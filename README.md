# Handwritten Digit Recognition System

This project is a Handwritten Digit Recognition System implemented in Python. It provides scripts to train a machine learning model on handwritten digits and to predict the digit from a given input image.

## Project Structure

- `train_model.py`: Script to build and train the digit recognition model.
- `predict_image.py`: Script to load the trained model and predict the digit from a given image.
- `digit_model.h5`: The trained model file (if generated).
- `my_digit.jpg`: A sample image for testing predictions.

## Setup and Usage

1. **Install dependencies**:
   Ensure you have the required libraries installed in your Python environment. You may need libraries such as `tensorflow` (or `keras`), `numpy`, `opencv-python` (cv2), and `matplotlib`.
   
   ```bash
   pip install tensorflow numpy opencv-python matplotlib
   ```

2. **Train the model**:
   If you need to train or retrain the model, run the training script. This will use the MNIST dataset by default to train a neural network and save it as `digit_model.h5`.
   ```bash
   python train_model.py
   ```

3. **Predict an image**:
   Use the prediction script to recognize a handwritten digit from an image file (e.g., `my_digit.jpg`).
   ```bash
   python predict_image.py
   ```
   *(Note: You may need to edit `predict_image.py` to point to the specific image path you want to predict if you are using a custom image).*

## Notes
- Images passed to the prediction script should ideally be preprocessed similarly to the training data (e.g., grayscale, 28x28 pixels, centered digit) for the best accuracy.
