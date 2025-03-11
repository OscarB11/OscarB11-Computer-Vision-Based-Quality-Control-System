# Sol-2: PCB Fault Detection using Deep Learning Classification

## Overview
Sol-2 provides a specialized image classification system designed to detect faults in PCB (Printed Circuit Board) images. Using TensorFlow-based deep learning models, this solution can automatically identify and categorize various PCB defects from image snippets, enabling efficient quality control in electronics manufacturing.

## Key Features
- **PCB-Specific Classification**: Detects common PCB defects including solder issues, component misalignment, and circuit breaks
- **High Accuracy**: >98% detection rate on common PCB defects
- **Optimized CNN Architecture**: Based on MobileNetV3-Large for efficient inference
- **Multi-class Fault Detection**: Categorizes defects into specific fault types for targeted remediation
- **Snippet Processing**: Analyzes small sections of PCBs for detailed inspection
- **Adjustable Confidence Thresholds**: Tune sensitivity based on manufacturing requirements

## Model Architecture
- Base network: MobileNetV3-Large optimized for PCB inspection
- Fine-tuning layers: Dense layers with dropout for PCB-specific feature learning
- Output: Multi-class classification of common PCB defects
- Training data: 50,000+ annotated PCB snippet images covering various fault conditions

## Pre-trained Model
The solution includes a pre-trained model optimized for PCB defect detection:

1. Download the model from our repository:
   [Download PCB Model (300MB)](https://drive.google.com/file/d/1Hg2QW7BJkm6Ei_5sLMTO-AcH7p6nIQJ3/view?usp=sharing)

2. Place the downloaded `.h5` file in the `models` directory

## Installation Requirements
- Python 3.8+
- TensorFlow 2.6+
- OpenCV 4.5+
- NumPy 1.20+
- Pillow 8.0+

## Quick Start
```python
from sol2_classifier import PCBClassifier

# Initialize with pretrained model
classifier = PCBClassifier(model_path="models/pcb_classifier_v2.h5")

# Classify a single PCB snippet image
result = classifier.classify("path/to/pcb_image.jpg")
print(f"Detected issue: {result['class']}, Confidence: {result['confidence']:.2f}")

# Batch classification of multiple PCB snippets
results = classifier.batch_classify(["pcb1.jpg", "pcb2.jpg", "pcb3.jpg"])
```

## Detectable PCB Defects
- Solder bridges
- Cold solder joints
- Missing components
- Misaligned components
- Copper exposure
- Circuit breaks
- Foreign material contamination
- Component damage

## Performance Metrics
- Inference time: ~50ms per snippet on CPU, ~15ms on GPU
- Memory footprint: 45MB (quantized model)
- Precision/Recall: 0.97/0.96 on PCB defect test datasets

## Integration Examples
The `examples` directory contains sample code for integration with:
- Automated inspection systems
- Manufacturing execution systems
- PCB production lines
- Highlight visualization tools

## License
This project is licensed under the MIT License.
