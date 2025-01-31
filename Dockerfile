# Use the official Python image as the base image
FROM python:3.12-slim


# Set the working directory in the container
WORKDIR /app

# Install g++ and other build essentials

RUN apt-get update && apt-get install -y \
    build-essential \
    cron \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++


# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

COPY cron/update_data.py /app/cron/

# Create the cron job
RUN echo "0 1 * * * /usr/local/bin/python /app/cron/update_data.py >> /var/log/cron.log 2>&1" > /etc/cron.d/update_job
RUN chmod 0644 /etc/cron.d/update_job
RUN crontab /etc/cron.d/update_job

# Create the log file to be able to run tail
RUN touch /var/log/cron.log

# Expose the port your app runs on (if it's a web app)
EXPOSE 5000

# Command to run your application
CMD service cron start && python app.py
