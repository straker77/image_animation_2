# Use an official Python runtime as a parent image
#FROM python:3.8-slim
FROM python:3.10.12
#FROM nvcr.io/nvidia/pytorch:21.02-py3
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime



# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

#COPY app.py /app
#COPY requirements.txt /app
#COPY sup-mat /app

# Update and install necessary libraries

#RUN apt-get update && apt-get install -y \
    #libsm6 \
    #libxext6 \
   # libxrender-dev \
   # libgl1-mesa-glx \
   # libglib2.0-0

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install python3-pip ffmpeg git less nano libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*    

# Upgrade pip
RUN pip3 install --upgrade pip
RUN pip3 install \
  git+https://github.com/1adrianb/face-alignment \
  -r requirements.txt 

# Install any needed packages specified in requirements.txt
#RUN pip install --upgrade pip && pip install -r requirements.txt



# Install PyTorch and torchvision
#RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Make port 80 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]



