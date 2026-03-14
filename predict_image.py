import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy import ndimage

model = keras.models.load_model("digit_model.h5")

img = cv2.imread("my_digit.jpg")

if img is None:
    print("Image not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Strong threshold using Otsu
_, thresh = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

largest = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest)

digit = thresh[y:y+h, x:x+w]

# Resize while preserving aspect ratio
if w > h:
    new_w = 20
    new_h = int(h * (20.0 / w))
else:
    new_h = 20
    new_w = int(w * (20.0 / h))

digit = cv2.resize(digit, (new_w, new_h))

# Create 28x28 canvas
canvas = np.zeros((28,28), dtype=np.uint8)

x_offset = (28 - new_w) // 2
y_offset = (28 - new_h) // 2

canvas[y_offset:y_offset+new_h,
       x_offset:x_offset+new_w] = digit

# Center using center of mass
cy, cx = ndimage.center_of_mass(canvas)
shiftx = np.round(14 - cx).astype(int)
shifty = np.round(14 - cy).astype(int)

canvas = ndimage.shift(canvas, (shifty, shiftx))

# Slight blur like MNIST
canvas = cv2.GaussianBlur(canvas, (3,3), 0)

# Normalize
canvas = canvas / 255.0
canvas = canvas.reshape(1,28,28,1)

prediction = model.predict(canvas)
predicted_digit = np.argmax(prediction)

print("Prediction probabilities:", prediction)
print("Predicted Digit:", predicted_digit)

plt.imshow(canvas.reshape(28,28), cmap='gray')
plt.title("Prediction: " + str(predicted_digit))
plt.show()