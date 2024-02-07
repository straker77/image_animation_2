import subprocess

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

if __name__ == "__main__":
    start_demo_script()








		



