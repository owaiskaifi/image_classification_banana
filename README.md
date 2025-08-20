# 🖼️ Image Classification Service

A production-ready image classification service using ResNet model with ONNX runtime, deployable via Docker and Banana serverless platform.

## 📋 Overview

This project provides a high-performance HTTP API for image classification using a pre-trained ResNet model. The service accepts images via POST requests and returns predicted class indices based on ImageNet categories.

### Key Features

- **Fast Inference**: ONNX runtime for optimized model inference
- **REST API**: Simple HTTP interface with JSON responses  
- **Containerized**: Docker support for easy deployment
- **Serverless Ready**: Compatible with Banana serverless platform
- **Production Ready**: Includes health checks and error handling

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │───▶│    Sanic    │───▶│    ONNX     │
│  (Images)   │    │   Server    │    │   Runtime   │
└─────────────┘    └─────────────┘    └─────────────┘
                          │                   │
                          │                   ▼
                          │            ┌─────────────┐
                          │            │   ResNet    │
                          │            │   Model     │
                          ▼            └─────────────┘
                   ┌─────────────┐
                   │ JSON Response│
                   │  {"class": N}│
                   └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker (optional)
- 4GB+ RAM for model loading

### Local Installation

1. **Clone the repository:**
```bash
git clone https://github.com/owaiskaifi/image_classification_banana.git
cd image_classification_banana
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download pre-trained model (if needed):**
```bash
python download.py
```

4. **Convert model to ONNX format:**
```bash
python convert_pytorch_model.py
```

5. **Start the server:**
```bash
python server.py
```

The service will be available at `http://localhost:8000`

### Docker Deployment

1. **Build the Docker image:**
```bash
docker build -t image-classifier .
```

2. **Run the container:**
```bash
docker run -p 8000:8000 image-classifier
```

## 📡 API Documentation

### POST `/`

Classify an image and return the predicted class index.

#### Request Format

```json
{
  "image": "<base64-encoded-image>"
}
```

#### Response Format

```json
{
  "class": 123
}
```

#### Example Usage

**Python:**
```python
import base64
import requests

# Load and encode image
with open("path/to/image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/",
    json={"image": image_data}
)

result = response.json()
print(f"Predicted class: {result['class']}")
```

**cURL:**
```bash
# First, encode your image to base64
IMAGE_B64=$(base64 -i image.jpg)

# Make the request
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_B64\"}"
```

## 🧪 Testing

### Local Testing

Test the service locally with sample images:

```bash
# Test with local server
python test.py
```

### Banana Deployment Testing

Test the deployed service on Banana platform:

```bash
# Update API keys in test_banana.py first
python test_banana.py
```

## 🏭 Deployment Options

### 1. Docker Deployment

The included Dockerfile creates a production-ready container:

```dockerfile
# Uses CUDA-enabled PyTorch base image
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Installs all dependencies and model files
# Exposes service on port 8000
```

### 2. Banana Serverless

Deploy to Banana's serverless platform:

1. **Follow the [Banana quickstart guide](https://docs.banana.dev/banana-docs/quickstart)**
2. **Choose "GitHub Repository" deployment method**
3. **Point to this repository**
4. **Update API keys in your test files**

### 3. Cloud Platforms

The Docker container can be deployed to any cloud platform that supports containers:

- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances
- Kubernetes clusters

## 📁 Project Structure

```
image_classification_banana/
├── server.py                 # Main HTTP server (Sanic-based)
├── convert_pytorch_model.py  # PyTorch to ONNX model conversion
├── download.py              # Pre-trained model downloader
├── test.py                  # Local testing script
├── test_banana.py           # Banana deployment testing
├── Dockerfile               # Container configuration
├── requirements.txt         # Python dependencies
├── resnet.onnx             # ONNX model file (generated)
├── n01440764_tench.jpeg    # Sample test image
├── n01667114_mud_turtle.JPEG # Sample test image  
├── cat.jpg                 # Sample test image
└── README.md               # This file
```

### File Descriptions

| File | Purpose |
|------|---------|
| `server.py` | Main HTTP server that loads ONNX model and serves predictions |
| `convert_pytorch_model.py` | Converts PyTorch ResNet model to ONNX format for inference |
| `download.py` | Downloads pre-trained ResNet50 weights from PyTorch Hub |
| `test.py` | Local testing script for development |
| `test_banana.py` | Testing script for Banana serverless deployment |
| `Dockerfile` | Container configuration for production deployment |
| `requirements.txt` | Python package dependencies |

## 🔧 Configuration

### Environment Variables

- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

### Model Configuration

The service uses ImageNet-pretrained ResNet with:
- Input size: 224x224 pixels
- Color format: RGB
- Normalization: ImageNet standard
- Output: 1000 ImageNet classes

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found**: Run `python convert_pytorch_model.py` to generate `resnet.onnx`
2. **CUDA out of memory**: Use CPU-only version or reduce batch size
3. **Import errors**: Install requirements with `pip install -r requirements.txt`
4. **Port conflicts**: Change port in server.py or use different port mapping in Docker

### Performance Optimization

- Use GPU-enabled container for faster inference
- Consider model quantization for reduced memory usage
- Implement batch processing for multiple images
- Add caching for frequently requested images

## 📊 Model Information

- **Architecture**: ResNet (Residual Neural Network)
- **Training Dataset**: ImageNet (1000 classes)
- **Input Resolution**: 224×224 pixels
- **Model Size**: ~25MB (ONNX format)
- **Inference Time**: ~50ms on CPU, ~5ms on GPU

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Useful Links

- [Banana Documentation](https://docs.banana.dev/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Sanic Framework](https://sanic.dev/)
- [PyTorch Hub](https://pytorch.org/hub/)

---

**Built for production image classification with ❤️**
