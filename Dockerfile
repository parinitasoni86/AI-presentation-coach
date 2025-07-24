# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install git and ffmpeg
RUN apt-get update && apt-get install -y git ffmpeg && apt-get clean

# Copy project files into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

