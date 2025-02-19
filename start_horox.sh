#!/bin/bash

# Change to the project directory
cd /home/thikka/projects/horox

# Activate the virtual environment
source /home/thikka/projects/horox/venv/bin/activate

# Start the Flask application with Gunicorn (add logging for debugging)
gunicorn --daemon --workers 4 --bind 0.0.0.0:5000 main:app >> /home/thikka/projects/horox/app_startup.log 2>&1

echo "Horox application started."