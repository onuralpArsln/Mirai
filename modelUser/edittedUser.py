from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
try:
    model = load_model("keras_model.h5", compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load the labels
try:
    with open("labels.txt", "r") as file:
        class_names = file.readlines()
except FileNotFoundError:
    print("Error: labels.txt file not found.")
    exit(1)

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image_path = "sad.png"
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"Error: {image_path} file not found.")
    exit(1)

# Resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predict the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:])
print("Confidence Score:", confidence_score)
