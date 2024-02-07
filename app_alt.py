import os  # Add this line to import the os module
import ssl
import cv2
import numpy as np
import subprocess

from flask import Flask, send_file, request , jsonify
#from werkzeug import secure_filename
from werkzeug.utils import secure_filename
from PIL import Image




import sys
import yaml
from argparse import ArgumentParser
from tqdm.auto import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp

import ffmpeg
from os.path import splitext
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

import logging
import time

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/send_image_1', methods=['GET', 'POST'])
def send_image_1():
    try:
        # Assuming the image file is named 'preexisting_image.jpg'
        image_path = 'cybill.jpg'
        
        # Perform image resizing logic here if needed
        
        # Send the resized image back in the response
        return send_file(image_path, mimetype='image/jpeg')

    except Exception as e:
        return str(e)

@app.route('/send_image_2', methods=['GET', 'POST'])
def send_image_2():
    try:
        # Assuming the image file is named 'preexisting_image.jpg'
        image_path = 'monika.png'
        
        # Perform image resizing logic here if needed
        
        # Send the resized image back in the response
        return send_file(image_path, mimetype='image/png')

    except Exception as e:
        return str(e)



# Ensure the 'uploads' folder exists
uploads_folder = os.path.join(os.getcwd(), 'uploads')
os.makedirs(uploads_folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = uploads_folder   




@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   global uploaded_filename
   if request.method == 'POST':
      f = request.files['file']
      uploaded_filename=secure_filename(f.filename)
     # uploaded_filepath='uploads/hure.png'

      filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
      f.save(filepath)
     # return 'file uploaded successfully'
      return filepath




@app.route('/reload/<timestamp>', methods=['GET', 'POST'])
def reload_file(timestamp):
    global uploaded_filename
    try:
       # f = request.files['file']
        # Assuming the image file is named 'preexisting_image.jpg'
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
       # image_path = 'marine.jpg'
        # Perform image resizing logic here if needed
        #uploaded_filepath='uploads/hure.png'
        
        # Send the resized image back in the response
        return send_file(image_path, mimetype='image/jpg')

    except Exception as e:
        return str(e)  

@app.route('/load_video/<timestamp>', methods=['GET', 'POST'])
def load_video(timestamp):
    global uploaded_filename
    try:
       # f = request.files['file']
        # Assuming the image file is named 'preexisting_image.jpg'
        logger.info('load_XXXX')
        image_path = 'result.mp4' #os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
        generate_video()
       # start_demo_script()
        logger.info('load_yu')
       # image_path = 'marine.jpg'
        # Perform image resizing logic here if needed
        #uploaded_filepath='uploads/hure.png'
        
        # Send the resized image back in the response
        return send_file(image_path, mimetype='video/mp4')

    except Exception as e:
        return str(e)                  

@app.route('/upload2/<sessionid>', methods = ['GET', 'POST'])
def upload_file2(sessionid):
   global uploaded_filename
   if request.method == 'POST':
      f = request.files['file']
      uploaded_filename=generate_filename_with_sessionid(f.filename, sessionid)
    
     

     # filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
      filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
      f.save(filepath)
     # return 'file uploaded successfully'
      return filepath      
                 

@app.route('/convert_to_png/<timestamp>', methods=['GET', 'POST'])
def convert_to_png(timestamp):
    global uploaded_filename
    try:
       
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)

            # Perform image resizing logic here if needed

            # Convert to PNG
            png_path = os.path.splitext(image_path)[0] + '.png'
            img = Image.open(image_path)
            img.save(png_path, format='PNG')

            # Send the resized image back in the response
            return send_file(image_path, mimetype='image/png')
           

    except Exception as e:
        return str(e)



@app.route('/convert_to_pdf/<timestamp>', methods=['GET', 'POST'])
def convert_to_pdf(timestamp):
    global uploaded_filename
    try:
       
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)

            # Perform image resizing logic here if needed

            # Convert to PNG
            pdf_path = os.path.splitext(image_path)[0] + '.pdf'
            img = Image.open(image_path)
            img.save(pdf_path, format='PDF')

            # Send the resized image back in the response
            return send_file(pdf_path, mimetype='application/pdf')
           

    except Exception as e:
        return str(e)

@app.route('/convert_to_jpg/<timestamp>', methods=['GET', 'POST'])
def convert_to_jpg(timestamp):
    global uploaded_filename
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)

        # Perform image resizing logic here if needed

        # Convert to JPG
        jpg_path = os.path.splitext(image_path)[0] + '.jpg'
        img = Image.open(image_path)
        # Convert to RGB mode if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save as JPEG
        img.save(jpg_path, format='JPEG')

        # Check if the file is created and has a non-zero size
        if os.path.exists(jpg_path):
            file_size = os.path.getsize(jpg_path)
            if file_size > 0:
                # Send the resized image back in the response
                return send_file(jpg_path, mimetype='image/jpeg')
            else:
                return f"Error: The resulting JPG file has zero size. File path: {jpg_path}"
        else:
            return "Error: Failed to convert to JPG or the resulting file is not created."

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/convert_to_gif/<timestamp>', methods=['GET', 'POST'])
def convert_to_gif(timestamp):
    global uploaded_filename
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)

        # Perform image resizing logic here if needed

        # Convert to JPG
        jpg_path = os.path.splitext(image_path)[0] + '.jpg'
        img = Image.open(image_path)
        # Convert to RGB mode if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save as JPEG
        img.save(jpg_path, format='GIF')

        # Check if the file is created and has a non-zero size
        if os.path.exists(jpg_path):
            file_size = os.path.getsize(jpg_path)
            if file_size > 0:
                # Send the resized image back in the response
                return send_file(jpg_path, mimetype='image/gif')
            else:
                return f"Error: The resulting JPG file has zero size. File path: {jpg_path}"
        else:
            return "Error: Failed to convert to JPG or the resulting file is not created."

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/convert_to_bmp/<timestamp>', methods=['GET', 'POST'])
def convert_to_bmp(timestamp):
    global uploaded_filename
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)

        # Convert to BMP
        bmp_path = os.path.splitext(image_path)[0] + '.bmp'
        img = Image.open(image_path)

        # Convert to RGB mode if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

         # Save as BMP with specific settings
        img.save(bmp_path, format='BMP', quality=100, compression=None)

        # Check if the file is created and has a non-zero size
        if os.path.exists(bmp_path):
            file_size = os.path.getsize(bmp_path)
            if file_size > 0:
                # Send the resized image back in the response
                return send_file(bmp_path, mimetype='image/bmp')
            else:
                return f"Error: The resulting BMP file has zero size. File path: {bmp_path}"
        else:
            return "Error: Failed to convert to BMP or the resulting file is not created."

    except Exception as e:
        return f"Error: {str(e)}"      



@app.route('/thermalvision/<timestamp>', methods=['GET', 'POST'])
def thermalvision(timestamp):
    global uploaded_filename
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)

        # Convert to JPG
        thermal_path = os.path.splitext(image_path)[0] + '_thermal.jpg'
        # Read the input image
        img = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply a color map to the grayscale image (using the 'COLORMAP_JET' for a thermal effect)
        thermal_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        # Save the thermal vision effect image
        cv2.imwrite(thermal_path, thermal_img)


         # Check if the file is created and has a non-zero size
        if os.path.exists(image_path):
            file_size = os.path.getsize(image_path)
            if file_size > 0:
                # Send the resized image back in the response
                return send_file(thermal_path, mimetype='image/jpg')
            else:
                return f"Error: The resulting thermal file has zero size. File path: {thermal_path}"
        else:
            return "Error: Failed to convert to thermal or the resulting file is not created."

    except Exception as e:
        return f"Error: {str(e)}"      


def generate_filename_with_sessionid(filename, sessionid):
    # Get the file extension
    file_extension = os.path.splitext(filename)[1]
    
    # Create a new filename with sessionid
    new_filename = f"{secure_filename(filename.replace(file_extension, ''))}_{sessionid}{file_extension}"

    return new_filename






def load_checkpoints(config_path, checkpoint_path, cpu=True):

    with open(config_path) as f:
        config = yaml.full_load(f)
    logger.info('Generating Video 5')
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    logger.info('Generating Video 5a')                                    
    if not cpu:
        generator.cuda()
    logger.info('Generating Video 6')
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    logger.info('Generating Video 7')
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
    logger.info('Generating Video 8')
    generator.load_state_dict(checkpoint['generator'])
    logger.info('Generating Video 9')
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    logger.info('Generating Video 10')
    if not cpu:
        generator = DataParallelWithCallback(generator)
        logger.info('Generating Video 11')
        kp_detector = DataParallelWithCallback(kp_detector)
        logger.info('Generating Video 12')
    generator.eval()
    logger.info('Generating Video 13')
    kp_detector.eval()
    logger.info('Generating Video 14')
    return generator, kp_detector
  

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment  # type: ignore (local file)
    from scipy.spatial import ConvexHull

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

   

def generate_video():
    global uploaded_filename
    image_path = "{}{}".format('uploads/',uploaded_filename)
    logger.info(image_path)
    logger.info('Generating Video test')
    source_image = imageio.imread(image_path) # imageio.imread('sup-mat/hitler.jpg')
    reader = imageio.get_reader('sup-mat/unknown_man_2.mp4')
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    logger.info('Generating Video 2')
    source_image = resize(source_image, (256, 256))[..., :3]
    logger.info('Generating Video 3')
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    logger.info('Generating Video 4')
    generator, kp_detector = load_checkpoints(config_path='config/vox-adv-256.yaml', checkpoint_path='vox-adv-cpk.pth.tar', cpu=False)

    parser = ArgumentParser()

    parser.add_argument("--config", default='config/vox-adv-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='vox-adv-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/hitler.jpg', help="path to source image")
    parser.add_argument("--driving_video", default='unknown_man_2.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")

    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None, help="Set frame to start from.")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    parser.add_argument("--audio", dest="audio", action="store_true", help="copy audio to output from the driving video" )

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)
    parser.set_defaults(audio_on=False)

    opt = parser.parse_args()

    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=False)
        print("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=False, adapt_movement_scale=False, cpu=False)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=False, adapt_movement_scale=False, cpu=False)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=False, adapt_movement_scale=False, cpu=False)
    imageio.mimsave('result.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)

    """
    if opt.audio:
        try:
            with NamedTemporaryFile(suffix=splitext(opt.result_video)[1]) as output:
                ffmpeg.output(ffmpeg.input(opt.result_video).video, ffmpeg.input(opt.driving_video).audio, output.name, c='copy').run()
                with open(opt.result_video, 'wb') as result:
                    copyfileobj(output, result)
        except ffmpeg.Error:
            print("Failed to copy audio: the driving video may have no audio track or the audio format is invalid.")
        """       
    return

    def start_demo_script():
    # Command to start the demo.py script with the specified parameters
        command = [
            "python",
            "demo.py",
            "--config",
            "config/vox-adv-256.yaml",
            "--driving_video",
            "sup-mat/unknown_man_2.mp4",
            "--source_image",
            "sup-mat/steinmeier.jpg",
            "--checkpoint",
            "vox-adv-cpk.pth.tar",
            "--relative",
            "--adapt_scale"
        ]

        # Start the demo.py script
        subprocess.run(command)        
 
               


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)







		



