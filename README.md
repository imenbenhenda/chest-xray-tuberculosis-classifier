# 🧠 Tuberculosis Detection using CNN and Grad-CAM

## 🎯 Objective
This project implements a deep learning pipeline to detect tuberculosis (TB) from chest X-ray (CXR) images using a custom Convolutional Neural Network (CNN) and explainable AI techniques.

### Key components:
1. **CNN model** trained to classify chest X-rays as either **Normal** or **Tuberculosis**
2. **Grad-CAM visualizations** to highlight the lung regions influencing the model's predictions
3. (Optional) **Web interface** for interactive diagnosis (Flask integration planned)

---

## 📁 Dataset
- **Source:** [TB Chest Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- **Total Images:** 4200
  - **Normal:** 3500
  - **Tuberculosis:** 700
- **Preprocessing:**
  - Resized to **150×150** pixels
  - Pixel normalization to `[0, 1]`
  - Data augmentation applied to training set (rotation, zoom, shift, flip)

---

## ⚖️ Handling Class Imbalance
The dataset is imbalanced (Normal: 3500, Tuberculosis: 700). To address this:

- **Class Weights:** Computed using `sklearn.utils.class_weight` and passed to the model during training to give more importance to the minority class.
- **Data Augmentation:** Applied to the training set to increase diversity and reduce overfitting on the dominant class.

These strategies helped the model achieve high recall on tuberculosis cases, which is critical in medical diagnostics.

---

## 🧱 Model Architecture
- **Type:** Custom CNN built with TensorFlow/Keras
- **Input Shape:** `(150, 150, 3)`
- **Layers:**
  1. `Conv2D(32)` → `BatchNormalization` → `MaxPooling2D`
  2. `Conv2D(64)` → `BatchNormalization` → `MaxPooling2D`
  3. `Conv2D(128)` → `BatchNormalization` → `MaxPooling2D`
  4. `Flatten`
  5. `Dense(128, relu)` → `Dropout(0.5)`
  6. `Dense(1, sigmoid)`
- **Optimizer:** `Adam` with learning rate `0.0001`
- **Loss Function:** `binary_crossentropy`
- **Metrics:** Accuracy, Precision, Recall, AUC

---

## 📊 Final Results
After training for 10 epochs with callbacks (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`), the model achieved:

| Metric               | Value     |
|----------------------|-----------|
| **Validation Accuracy** | **98.0%** |
| **Validation Recall**   | **97.1%** |
| **Validation Precision**| **86.1%** |
| **Validation AUC**      | **99.6%** |
| **Validation Loss**     | **0.0992** |

Note: The original version of this project achieved 97% validation accuracy. After further optimization, the final model reached 98% accuracy with stronger recall and AUC.

---

## 🔍 Explainable AI (Grad-CAM)
To interpret the model's decisions, Grad-CAM heatmaps are generated for validation images. These visualizations highlight the lung regions that influenced the prediction, helping build trust in the model’s output.

- Implemented using TensorFlow and OpenCV
- Displays both true and predicted labels with confidence scores
- Heatmaps are overlaid on original X-ray images

---

## ⚙️ Technologies Used
- Python
- TensorFlow & Keras
- NumPy & Matplotlib
- OpenCV (Grad-CAM)
- Scikit-learn (metrics)
- (Optional) Flask for web deployment

---

## 📂 Project Structure
tuberculosis-detection/
├ ── notebook/                       
├      ├── explainability_gradcam.ipynb  # Grad-CAM visualization notebook
├      ├── training.ipynb              # Model training notebook
├            
├── models/                     # Saved trained models (.keras format)
│
├── data/                       # Chest X-ray dataset (TB_Chest_Radiography_Database)
│
├── templates/                  # HTML templates for the web interface
│   └── index.html              # Main page for image upload and prediction
│── README.md                   # Project documentation
│── requirements.txt            # Python dependencies
│── app.py                      # Flask web application for interactive diagnosis
  
---

## 👩‍💻 Author
**Imen Ben Henda**  
Computer Engineering Student  
Focused on AI for healthcare and model interpretability