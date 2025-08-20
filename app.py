from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import cv2
import tensorflow as tf
app = Flask(__name__)

# Charge ton modèle entraîné
MODEL_PATH = os.path.join('model', 'tb_cnn_model.h5')
model = load_model(MODEL_PATH)
# Fonction pour trouver la dernière couche Conv2D
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Pas de couche Conv2D trouvée.")

# Fonction Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)

    # Créer un modèle fonctionnel temporaire
    inputs = tf.keras.Input(shape=img_array.shape[1:])
    x = inputs
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x
    output = x
    grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_output, output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = 0
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap

# Fonction pour superposer la heatmap
def overlay_heatmap(heatmap, img_bgr, alpha=0.35):
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlaid = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1-alpha, 0)
    return overlaid
def predict_tb(image_path):
    # Charge et prépare l'image
    image = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction
    prediction = model.predict(img_array)[0][0]

    # Générer Grad-CAM
    last_conv_layer_name = find_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Image originale en BGR
    img_bgr = cv2.cvtColor((img_array[0]*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    overlaid = overlay_heatmap(heatmap, img_bgr)

    # Sauvegarder image Grad-CAM
    gradcam_path = os.path.join('static', 'gradcam_'+os.path.basename(image_path))
    cv2.imwrite(gradcam_path, overlaid)

    return prediction, gradcam_path

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    gradcam_image = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join('static', file.filename)
            file.save(filepath)

            # Prédiction + Grad-CAM
            pred, gradcam_image = predict_tb(filepath)
            result = "Tuberculose détectée ⚠️" if pred >= 0.5 else "Poumons normaux ✅"

            return render_template('index.html', result=result, image=filepath, gradcam=gradcam_image)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
