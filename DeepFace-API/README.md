# DeepFace API - Face Recognition & Analysis

A comprehensive face recognition and analysis web application built with FastAPI, DeepFace, and a modern web interface. This application provides face verification and detailed facial analysis including age, gender, and emotion detection.

## Features

- **Face Verification**: Compare two faces to determine if they belong to the same person
- **Face Analysis**: Analyze facial attributes including:
  - Age estimation
  - Gender classification
  - Emotion detection (happy, sad, angry, surprise, fear, disgust, neutral)
- **Modern Web Interface**: GitHub-style dark theme with responsive design
- **Docker Support**: Fully containerized application with Docker Compose
- **REST API**: FastAPI backend with comprehensive API documentation

## Tech Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **DeepFace**: Deep learning library for face recognition and analysis
- **TensorFlow-CPU**: Machine learning framework
- **Python 3.9+**: Programming language

### Frontend
- **HTML5/CSS3**: Modern web standards
- **Bootstrap 5.3.0**: Responsive CSS framework with dark theme support
- **Vanilla JavaScript**: Client-side functionality
- **Nginx**: Web server for serving static files

### Infrastructure
- **Docker**: Containerization platform
- **Docker Compose**: Multi-container orchestration

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Git (for cloning the repository)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd deep_face
```

2. Start the application:
```bash
docker-compose up -d
```

3. Access the application:
- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Redoc**: http://localhost:8000/redoc

## Usage

### Web Interface

1. **Face Verification**:
   - Upload two images in the "Face Verification" tab
   - Click "Verify Faces" to compare them
   - View the verification result and confidence score

2. **Face Analysis**:
   - Upload an image in the "Face Analysis" tab
   - Click "Analyze Face" to get detailed analysis
   - View age, gender, and emotion predictions

### API Endpoints

#### Face Verification
```http
POST /verify
Content-Type: application/json

{
  "image1": "base64_encoded_image",
  "image2": "base64_encoded_image"
}
```

#### Face Analysis
```http
POST /analyze
Content-Type: application/json

{
  "image": "base64_encoded_image"
}
```

## Project Structure

```
deep_face/
├── deepface/                 # Backend service
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Backend Docker configuration
├── frontend/                 # Frontend service
│   ├── index.html           # Web interface
│   ├── app.js              # JavaScript functionality
│   ├── nginx.conf          # Nginx configuration
│   └── Dockerfile          # Frontend Docker configuration
├── docker-compose.yaml      # Multi-container orchestration
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

## Development

### Backend Development

The backend is built with FastAPI and uses the DeepFace library for face recognition:

- **main.py**: Contains the FastAPI application with verification and analysis endpoints
- **TensorFlow CPU**: Configured to suppress CUDA warnings for CPU-only deployment
- **CORS**: Enabled for cross-origin requests from the frontend

### Frontend Development

The frontend is a modern web application with a GitHub-style dark theme:

- **Responsive Design**: Works on desktop and mobile devices
- **Dark Theme**: Professional GitHub-inspired color scheme
- **Drag & Drop**: Intuitive image upload interface
- **Real-time Results**: Dynamic display of analysis results

### Environment Variables

The application uses environment variables to suppress TensorFlow warnings:

```bash
TF_CPP_MIN_LOG_LEVEL=2
TF_ENABLE_ONEDNN_OPTS=0
```

## API Documentation

Once the application is running, you can access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive documentation for all API endpoints.

## Docker Configuration

### Backend Service (deepface)
- **Port**: 8000
- **Image**: Built from `./deepface/Dockerfile`
- **Dependencies**: Python 3.9, FastAPI, DeepFace, TensorFlow-CPU

### Frontend Service (frontend)
- **Port**: 3000
- **Image**: Built from `./frontend/Dockerfile`
- **Server**: Nginx Alpine
- **Files**: Serves static HTML, CSS, and JavaScript

## Troubleshooting

### Common Issues

1. **TensorFlow Warnings**: The application is configured to suppress CUDA warnings when running on CPU
2. **Port Conflicts**: Ensure ports 3000 and 8000 are available
3. **Docker Issues**: Try rebuilding containers with `docker-compose build --no-cache`

### Performance Notes

- Face analysis may take a few seconds depending on image size and system resources
- The application uses TensorFlow-CPU for broad compatibility
- For better performance in production, consider using TensorFlow-GPU with appropriate hardware

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) - Deep learning library for face recognition
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework for building APIs
- [Bootstrap](https://getbootstrap.com/) - CSS framework for responsive design
- [TensorFlow](https://www.tensorflow.org/) - Machine learning platform

## Contact

For questions or support, please open an issue in the GitHub repository.
