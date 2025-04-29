import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import json
import os

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python predict_model.py path_to_image.jpg")
        sys.exit(1)
    img_path = sys.argv[1]

    # Check if image exists
    if not os.path.isfile(img_path):
        print(f"Error: File '{img_path}' does not exist.")
        sys.exit(1)

    # Load model
    model = tf.keras.models.load_model('save_model.h5')

    # Load class names
    with open('class_names.json', 'r') as f:
        class_labels = json.load(f)

    # Load & preprocess image (must match training image size!)
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    pred = model.predict(x)
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred))

    print(f"Predicted Class: {class_labels[class_idx]}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main()
