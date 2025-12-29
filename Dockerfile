# Use an official Python runtime as a parent image
# python:3.9-slim is a good balance of size and compatibility
FROM python:3.9-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV
# libgl1-mesa-glx: Required for cv2
# libglib2.0-0: Required for cv2
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
# If you don't have a requirements.txt generated yet, we use pyproject.toml or install manually
# For this Dockerfile we'll assume pip usage for simplicity or use the pyproject.toml
RUN pip install --no-cache-dir opencv-python numpy ultralytics

# Copy project files
COPY fall_detector.py .
COPY README.md .

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Document that this app needs a display/webcam access
# Note: To run this container with webcam access and display, use:
# docker run -it --rm --device=/dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix fall-detector
CMD ["python", "fall_detector.py", "--no-audio"]
