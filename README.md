# 🤟 Sign Language Detection

An **AI-powered Sign Language Detection application** that recognizes and interprets hand gestures in real time using **Deep Learning and Computer Vision**. The application converts detected sign language gestures into readable text, helping bridge the communication gap between the hearing-impaired community and others.

## 🧩 Overview

This project uses a trained **MobileNetV2-based Convolutional Neural Network (CNN)** model to classify hand gestures.

The model is integrated with a **Flask web application** that provides an interactive interface for real-time gesture recognition through a webcam. Hand detection and preprocessing are performed using **MediaPipe and OpenCV**, while the frontend captures webcam frames and sends them to the backend for prediction.

## ⚙️ Features

- 🎥 Real-time sign language detection using a webcam
- 🧠 Deep learning-based gesture classification using MobileNetV2
- ✋ Hand detection and tracking using MediaPipe
- 🖼️ Image preprocessing using OpenCV
- 🌐 Flask-powered web application
- 💬 Converts recognized gestures into readable text
- ⚡ Prediction smoothing for more stable results
- 🧹 Interactive controls to manage the recognized sentence
- 📱 Browser-based webcam access

## 🧠 Tech Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Backend Framework | Flask |
| Deep Learning | TensorFlow / Keras |
| Model Architecture | MobileNetV2 |
| Computer Vision | OpenCV |
| Hand Detection | MediaPipe |
| Data Processing | NumPy |
| Machine Learning Utilities | Scikit-learn, Joblib |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render, Gunicorn |

## 🏗️ System Architecture

1. **Webcam Capture:** The browser captures live video using the webcam.
2. **Frame Processing:** Captured frames are sent to the Flask backend.
3. **Hand Detection:** MediaPipe detects the hand landmarks and bounding region.
4. **Image Preprocessing:** OpenCV extracts and preprocesses the hand region.
5. **Gesture Classification:** The trained MobileNetV2 model predicts the gesture.
6. **Prediction Stabilization:** Multiple predictions are used to improve stability.
7. **Text Output:** The recognized gesture is displayed and combined into a sentence.

## 📂 Project Structure

```text
Indian-Sign-Language/
│
├── app.py
├── requirements.txt
├── Procfile
├── .python-version
│
├── dataset/
│   ├── label_encoder.pkl
│   └── sign_language_mobilenet.h5
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
└── templates/
    └── index.html
```

## 🔮 Future Improvements

- Support for additional sign language gestures
- Sentence formation with improved language processing
- Improved prediction accuracy under different lighting conditions
- Support for multiple hand gestures
- Mobile-friendly experience
- Voice output for recognized text

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

## 📄 License

This project is available for educational and research purposes.

## 📸 Demo Preview
<img width="1920" height="1080" alt="demo_collage" src="https://github.com/user-attachments/assets/5af1c11e-0058-4bc2-bdef-baf3c7f99115" />




