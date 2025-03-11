# Sol-3: Object Detection Quality Control System with .NET Integration

## Overview
Sol-3 is a comprehensive quality control system that uses object detection models to identify and locate defects in manufacturing products. The system combines powerful deep learning models with a Flask server backend and a Windows .NET desktop application frontend, creating an end-to-end solution for visual inspection in production environments.

## Core Capabilities
- **Precision Defect Detection**: Identifies defects with exact location and boundaries
- **Multi-class Classification**: Recognizes and categorizes multiple defect types simultaneously
- **Real-time Processing**: Achieves production-ready processing speeds
- **.NET Windows Application**: User-friendly desktop interface for production staff
- **Flask Backend Server**: Handles model inference and API communications
- **Comprehensive Reporting**: Generates detailed inspection reports with defect visualization

## Supported Detection Models
1. **General-purpose Models**:
   - SSD MobileNet V2: Balanced speed/accuracy for general defect detection
   - EfficientDet: Higher accuracy for complex scenes
   - YOLO v5: Fast detection with good accuracy

2. **Manufacturing-specific Models**:
   - PCB Defect Detector: Specialized for electronics manufacturing
   - Surface Defect Detector: For metal, plastic, and painted surfaces
   - Assembly Verification: Ensures correct component assembly

## Technical Architecture
- **Backend**: Flask server running optimized TensorFlow/PyTorch models
- **Frontend**: .NET Windows desktop application
- **Communication**: REST API between frontend and backend
- **Processing Pipeline**: Image acquisition → preprocessing → detection → result visualization
- **Storage**: Local database for results and inspection history

## Requirements
- **Backend**:
  - Python 3.8+
  - TensorFlow 2.5+ or PyTorch 1.9+
  - Flask 2.0+
  - OpenCV 4.5+
  - NumPy 1.20+

- **Frontend**:
  - Windows 10/11
  - .NET Framework 4.7+ or .NET 6.0+
  - Visual C++ Redistributable

## Installation
```bash
# Clone repository
git clone https://github.com/OscarB11/Computer-Vision-Based-Quality-Control-System.git
cd "Sol-3 object detection"

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r requirements.txt

# Download detection models
python download_models.py

# Start the Flask server
python flask_server.py
```

After starting the Flask server, launch the .NET desktop application and connect to the server.

## System Workflow
1. User selects inspection type through .NET desktop interface
2. Application captures or loads product images
3. Images are sent to Flask backend for processing
4. Object detection models identify defects and their locations
5. Results are returned to the .NET application
6. Defects are visualized on the original image with bounding boxes
7. Inspection reports are generated and stored

## Performance Benchmarks
| Model | Speed (ms/image) | Accuracy | Use Case |
|-------|-----------------|----------|----------|
| MobileNet SSD | 45ms | 91.2% | General inspection |
| EfficientDet | 120ms | 95.8% | Precision inspection |
| YOLO v5 | 25ms | 93.5% | High-speed lines |

## Key Features of the .NET Application
- Live camera feed integration
- Batch processing capability
- Defect visualization tools
- Inspection history tracking
- Configurable detection parameters
- User access management
- Exportable reports
