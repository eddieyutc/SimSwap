# FROM 763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu118-ubuntu20.04-ec2
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app
COPY . .

RUN pip install opencv-python pillow numpy==1.23.5 moviepy onnxruntime insightface==0.2.1 timm==0.5.4 imageio==2.4.1 fastapi "uvicorn[standard]" python-multipart

EXPOSE 8000
CMD ["uvicorn", "--host=0.0.0.0", "--port=8000", "main:app"]
