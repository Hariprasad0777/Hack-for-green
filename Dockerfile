# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the source code and configuration
COPY pyproject.toml .
COPY src/ src/

# Install the package in production mode
RUN pip install --no-cache-dir .

# Copy remaining files (README, etc)
COPY . .

# Default command: Launch the reactive pipeline via CLI
CMD ["offroad-stream"]
