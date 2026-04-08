# 😷 Face Mask Detection using Deep Learning

Real-time face mask detection system built with MobileNetV2 (Transfer Learning), TensorFlow, and OpenCV.

## 🎯 Problem Statement
Automating face mask compliance monitoring in public spaces using computer vision.

## 🛠️ Tech Stack
- Python 3.11
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- OpenCV
- Streamlit
- scikit-learn

## 📁 Project Structure
face-mask-detection/
├── model/                    # Saved training plot
├── src/
│   ├── train.py              # Train the model
│   ├── detect_webcam.py      # Real-time webcam detection
│   └── app.py                # Streamlit web app
├── requirements.txt
└── README.md

## 🚀 Getting Started
git clone https://github.com/Pritam-13/face-mask-detection.git
cd face-mask-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## 📦 Dataset
Download from Kaggle: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
Place images in dataset/with_mask/ and dataset/without_mask/

## ▶️ Run
Train model:
python3 src/train.py

Webcam detection:
python3 src/detect_webcam.py

Web app:
streamlit run src/app.py

## 📊 Results
- Model accuracy: ~98% on test set
- Real-time detection with green (mask) / red (no mask) bounding boxes
