from waitress import serve
from app import app  # Import your Flask app variable

if __name__ == '__main__':
    print("Server is running on http://localhost:8080")
    serve(app, host='0.0.0.0', port=8080, threads=6)