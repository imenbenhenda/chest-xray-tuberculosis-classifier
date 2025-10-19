# Tuberculosis Detection using CNN and Explainable AI

## Objective
This project develops a complete Deep Learning pipeline for detecting tuberculosis (TB) from chest X-ray (CXR) images.

This system includes:
1.  **A Convolutional Neural Network (CNN)** trained to classify X-rays as 'Normal' or 'Tuberculosis'.
2.  **Explainable AI (XAI)** using **Grad-CAM** to visualize *why* the model makes a specific decision.
3.  **A Flask Web Application** providing an interactive interface for users to upload an X-ray and receive an instant diagnosis and heatmap.

## Dataset
-   **Source:** [TB Chest Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
-   **Description:** A collection of 4200 chest X-ray images, split into 3500 'Normal' and 700 'Tuberculosis' cases.
-   **Preprocessing:**
    -   Resizing images to **(150, 150)** pixels.
    -   Normalization of pixel values (rescale to `[0, 1]`).
    -   Data Augmentation (rotation, zoom, flip) applied to the training set to prevent overfitting.

## Model Architecture
-   **Type:** Custom Convolutional Neural Network (CNN)
-   **Input Shape:** `(150, 150, 3)`
-   **Structure:**
    1.  `Conv2D(32)` + `BatchNormalization` + `MaxPooling2D`
    2.  `Conv2D(64)` + `BatchNormalization` + `MaxPooling2D`
    3.  `Conv2D(128)` + `BatchNormalization` + `MaxPooling2D`
    4.  `Flatten`
    5.  `Dense(128, activation='relu')`
    6.  `Dropout(0.5)`
    7.  `Dense(1, activation='sigmoid')`
-   **Optimizer:** `Adam(learning_rate=0.0001)`
-   **Loss Function:** `binary_crossentropy`

## Final Results
The model was trained using callbacks for `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau`, achieving the following performance on the validation set:

| Metric | Value |
| :--- | :--- |
| **Validation Accuracy** | **98.69%** |
| **Validation Loss** | **0.0360** |

These stable and high-performance results were achieved after optimizing the image size and learning rate.

## Key Features
-   🏥 **High Accuracy:** Achieves **98.7%** accuracy in classifying tuberculosis, providing a reliable diagnostic aid.
-   🔍 **Explainable AI (XAI):** Integrates Grad-CAM to produce heatmaps, showing exactly which parts of the lung the model focused on for its prediction.
-   🌐 **Interactive Demo:** A user-friendly web application built with Flask allows for easy testing by uploading an image.
-   🛠️ **Robust Training:** The model is optimized and stable, avoiding the common pitfalls of overfitting and instability.

## Technologies Used
-   Python
-   TensorFlow & Keras
-   Flask (Web Application)
-   OpenCV (for Grad-CAM)
-   NumPy
-   Matplotlib

## Project Structure

Project Structure
tuberculosis-detection/
├── src/
│   ├── train.py          # Model training script
│   ├── predict.py        # Inference script
│   └── grad_cam.py       # Explainability visualization
├── models/               # Trained models
├── data/                 # Dataset configuration
├── notebooks/            # Exploration and analysis
└── results/              # Output visualizations
Author
Imen Ben Henda - Computer Engineering Student

