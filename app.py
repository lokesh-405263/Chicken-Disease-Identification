import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import BatchNormalization

# Initialize Flask app
app = Flask(__name__)

# Define the class labels (assuming all models use the same class labels)
class_labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# Path to models
MODEL_PATHS = {
    "VGG16": "models/vgg_16.h5",
    "MobileNetV3": "models/mobilenetv3.h5",
    "EfficientNetB5": "models/efficient_net_B5.h5"
}

# Function to load the selected model
def load_selected_model(model_name):
    model_path = MODEL_PATHS.get(model_name)
    if model_path:
        try:
            # Use custom_objects to handle potential custom layers like BatchNormalization
            model = load_model(model_path, custom_objects={'BatchNormalization': BatchNormalization})
            return model
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return None
    else:
        return None

# Home route
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")  # Home page with an introduction

# Upload page route
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Get selected model from the form
        selected_model = request.form.get("model")

        if not selected_model:
            return jsonify({"error": "No model selected"}), 400

        try:
            # Load the selected model
            model = load_selected_model(selected_model)
            if not model:
                return jsonify({"error": "Model not found or error during loading"}), 400

            # Save the uploaded file
            filepath = os.path.join("uploads", file.filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(filepath)

            # Preprocess the image
            image_size = (128, 128)  # Use a consistent image size for all models
            img = load_img(filepath, target_size=image_size)
            img_array = img_to_array(img) / 255.0  # Normalize the image
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]

            # Cleanup the uploaded file
            os.remove(filepath)

            return render_template("result.html", prediction=predicted_class, model=selected_model)

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return render_template("upload.html")  # Upload page with form to upload and choose model

if __name__ == "__main__":
    app.run(debug=True)
