# Computer Vision: Cup and Pen Classifier

This project implements a deep learning computer vision model to classify images of cups and pens using neural networks.

## Project Overview

This binary classification project uses convolutional neural networks (CNNs) to distinguish between images of cups and pens. The model is trained on a custom dataset of cup and pen images and can predict the class of new images.

## Dataset

The image dataset contains:

- **Cup Images**: Located in `data/mydata/cups/` directory
- **Pen Images**: Located in `data/mydata/pens/` directory
- **Additional Data**: Various image collections in subdirectories

The dataset includes hundreds of images for each class, providing sufficient training data for the classification model.

## Model Architecture

The project uses:
- **Convolutional Neural Network (CNN)**: Deep learning model for image classification
- **Image Preprocessing**: Normalization and resizing of input images
- **Binary Classification**: Output layer with sigmoid activation for cup/pen prediction

## Files Structure

- `models/CV_model.py`: Model definition and training script
- `models/cup_pen_classifier.h5`: Saved trained model weights
- `data/`: Image dataset organized by class
  - `mydata/cups/`: Cup images for training
  - `mydata/pens/`: Pen images for training
  - Additional image collections for extended datasets

## Key Features

- Image data preprocessing and augmentation
- CNN model architecture with multiple layers
- Binary classification for cup vs pen detection
- Model training with validation split
- Saved model weights for future predictions
- Performance evaluation metrics

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages: tensorflow/keras, opencv-python, numpy, matplotlib, pillow

### Installation

1. Install dependencies:
   ```
   pip install tensorflow opencv-python numpy matplotlib pillow
   ```

2. Train the model:
   ```
   python models/CV_model.py
   ```

3. Load and use trained model:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('models/cup_pen_classifier.h5')
   ```

## Model Training

The training process includes:
1. **Data Loading**: Reading images from cup and pen directories
2. **Preprocessing**: Resizing images and normalizing pixel values
3. **Model Compilation**: Setting up the CNN architecture
4. **Training**: Fitting the model on the training data
5. **Validation**: Evaluating performance on test set
6. **Saving**: Storing trained model weights

## Usage

1. Prepare your image dataset in the appropriate directory structure
2. Run the training script to train the model
3. Use the saved model to predict new images:
   ```python
   prediction = model.predict(new_image)
   class_label = "Cup" if prediction > 0.5 else "Pen"
   ```

## Applications

This computer vision model can be used for:
- Object recognition and classification
- Automated inventory management
- Educational computer vision projects
- Building blocks for more complex vision systems
- Demonstrating CNN capabilities on real-world data

## Performance Metrics

The model provides:
- Training and validation accuracy
- Loss curves during training
- Confusion matrix for test predictions
- Classification report with precision and recall
