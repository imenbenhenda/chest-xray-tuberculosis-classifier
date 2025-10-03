# Tuberculosis Detection from Chest X-ray Images

## Objective
Develop a Deep Learning (CNN) model capable of detecting tuberculosis from chest radiograph images.  
The project includes explainability via **Grad-CAM** to visualize the lung regions that influence predictions.

## Dataset
[TB Chest Radiography Database](https://www.kaggle.com/datasets/nih-chest-xrays/data)  

- Normal and tuberculosis lung images
- Preprocessing: resizing to 224x224, pixel normalization

Dataset structure used:
TB_Chest_Radiography_Database/
├── Tuberculosis/
│ ├── img_0001.png
│ └── ...
├── Normal/
│ ├── img_0001.png
│ └── ...
├── Tuberculosis.metadata.xlsx
├── Normal.metadata.xlsx
└── README.md.txt
## Model Architecture
- CNN with multiple Conv2D + MaxPooling layers  
- Final Dense layer for binary classification (TB / Normal)  
- Activation: Sigmoid  
- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Explainability: Grad-CAM applied on the last convolutional layer

## Results
| Metric                 | Value |
|------------------------|-------|
| Accuracy               | ~92%  |
| Precision (TB)         | 93%   |
| Recall (TB)            | 88%   |
| F1-Score (TB)          | 90%   |

## Key Features
- 🏥 **Medical AI** - Automated TB detection from chest X-rays
- 🔍 **Explainable AI** - Grad-CAM visualization for interpretability
- 📊 **High Performance** - 92% accuracy in tuberculosis classification
- 🛠️ **Production Ready** - End-to-end deep learning pipeline

## Technologies Used
- Python
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Installation & Usage
```bash
# Clone repository
git clone https://github.com/your-username/tuberculosis-detection.git
cd tuberculosis-detection

# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py

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

