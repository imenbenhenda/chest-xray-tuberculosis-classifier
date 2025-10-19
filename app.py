# ==============================================================================
#                      FLASK WEB APPLICATION FOR TB DETECTION
# ==============================================================================

import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- 1. Initialization and Model Loading ---
app = Flask(__name__)

# Load the best trained model from the specified path
MODEL_PATH = os.path.join('model', 'tb_cnn_model_best.keras')
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# --- 2. Grad-CAM Helper Functions ---
def find_last_conv_layer(model):
    """Finds the name of the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generates a Grad-CAM heatmap using a robust method that manually
    reconstructs the model graph to ensure compatibility.
    """
    # Create a new model input
    inputs = tf.keras.Input(shape=img_array.shape[1:])
    
    # Manually reconstruct the graph by iterating through layers
    x = inputs
    conv_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x # Capture the output of the target conv layer
            
    # Create the Grad-CAM model with the reconstructed graph
    grad_model = tf.keras.Model(inputs, [conv_output, x])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]

    # Calculate gradients and the heatmap
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8) # Normalize
    return heatmap

def overlay_heatmap(heatmap, img_bgr, alpha=0.4):
    """Superimposes the heatmap on the original image."""
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)

# --- 3. Prediction Pipeline ---
def get_prediction_and_gradcam(image_path):
    """
    Processes an image to get a prediction and generate a Grad-CAM visualization.
    """
    # Load and preprocess the image
    image = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(image)
    img_array_scaled = img_array / 255.0
    img_array_expanded = np.expand_dims(img_array_scaled, axis=0)

    # Get model prediction
    prediction_score = model.predict(img_array_expanded)[0][0]

    # Generate Grad-CAM visualization
    last_conv_layer_name = find_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(img_array_expanded, model, last_conv_layer_name)
    
    # Overlay heatmap on the original image (in BGR format for OpenCV)
    img_bgr = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    overlaid_img = overlay_heatmap(heatmap, img_bgr)
    
    # Save the generated Grad-CAM image to the static folder
    gradcam_filename = 'gradcam_' + os.path.basename(image_path)
    gradcam_path = os.path.join('static', gradcam_filename)
    cv2.imwrite(gradcam_path, overlaid_img)
    
    return prediction_score, gradcam_path

# --- 4. Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the main page logic for image upload and prediction display."""
    result_text = None
    original_image_path = None
    gradcam_image_path = None

    if request.method == 'POST':
        # Check if an image file was uploaded
        file = request.files.get('image')
        if file and file.filename:
            # Save the uploaded file to the static directory
            filepath = os.path.join('static', file.filename)
            file.save(filepath)

            # Get prediction and Grad-CAM visualization
            pred_score, gradcam_image_path = get_prediction_and_gradcam(filepath)
            
            # Determine the result text based on the prediction score
            if pred_score >= 0.5:
                result_text = "Result: Tuberculosis Detected ⚠️"
            else:
                result_text = "Result: Normal ✅"
            
            original_image_path = filepath
            
    # Render the main page with the results
    return render_template('index.html', 
                           result=result_text, 
                           image=original_image_path, 
                           gradcam=gradcam_image_path)

# --- 5. Application Entry Point ---
if __name__ == '__main__':
    app.run(debug=True)