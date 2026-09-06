# 🫁 Tuberculosis Detection Flask API

A lightweight, high-performance Flask REST API powered by a fine-tuned **DenseNet** deep learning model (`densenet_tb_model.keras`). This service accepts chest X-ray images and classifies them into **Normal** or **Tuberculosis** with probability scores and clinical confidence levels.

Designed to serve as the AI microservice engine for full-stack (MERN) medical diagnostic applications.

---

## 🚀 Key Features

* **Optimized Inference:** Uses compiled C++ graph execution (`@tf.function`) to execute predictions in under 200ms on CPU.
* **Safety Thresholding:** Implements a strict medical safety threshold (`0.35`) to prioritize detection sensitivity for Tuberculosis screening.
* **Native In-Memory Preprocessing:** Decodes binary image data directly using TensorFlow C++ ops without disk I/O bottlenecks.
* **MERN-Ready & CORS Enabled:** Cross-Origin Resource Sharing is enabled out of the box to communicate with Node.js/Express or React frontends.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Framework:** Flask, Flask-CORS
* **Deep Learning Engine:** TensorFlow 2.x / Keras (DenseNet121 Architecture)
* **Image Processing:** TensorFlow Vision Engine, Pillow

---

## 📂 Project Structure

```text
Flask-API/
├── app.py                   # Main Flask application and model pipeline
├── densenet_tb_model.keras  # Trained DenseNet model file
├── requirements.txt         # Project dependencies
└── .gitignore               # Git untracked files setup
