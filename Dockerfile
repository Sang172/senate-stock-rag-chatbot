# Use the official Python image as the base image
FROM python:3.12-slim

# Install g++ and other build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port your app runs on (if it's a web app)
EXPOSE 5000

# Define environment variables for AWS credentials (best practice to pass these via ECS task definition or environment)
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_REGION=us-west-1
ENV S3_BUCKET=senate-stock-rag-chatbot

# Command to run your application
CMD ["python", "app.py"]