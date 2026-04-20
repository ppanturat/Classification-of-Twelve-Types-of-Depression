# Depression Classification App - Docker Setup

This Flask app classifies depression types using machine learning models.

## Prerequisites
- Docker installed on your system
- Docker Compose (optional, for easier management)

## Quick Start

### Using Docker Compose (Recommended)
1. Navigate to the `depression_app` directory
2. Run: `docker-compose up --build`
3. Open http://localhost:5000 in your browser

### Using Docker directly
1. Navigate to the `depression_app` directory
2. Build the image: `docker build -t depression-app .`
3. Run the container: `docker run -p 5000:5000 depression-app`
4. Open http://localhost:5000 in your browser

## Models
The app automatically loads `.pkl` model files from the `models/` folder at startup. The models are included statically in the Docker image.

## Development
For development with live reloading, modify the `docker-compose.yml` to mount the code volume and use debug mode.

## Production
For production deployment, consider using a WSGI server like Gunicorn instead of the Flask development server.