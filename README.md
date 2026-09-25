# 🫁 DenseNet121 Tuberculosis Diagnostic API

A high-performance, containerized Flask REST API for automated Tuberculosis (TB) screening from Chest X-rays using a fine-tuned **DenseNet121** deep learning model.

Designed for seamless microservice integration, low-latency execution, and cross-platform compatibility across Windows, macOS, and Linux via Docker.

---

## ⚡ Key Architecture & Performance Features

* **3-Tier Clinical Triage Logic:** Categorizes X-ray scans into actionable risk levels:
  * **Normal (`<25%` risk):** No immediate follow-up required.
  * **Examination Required (`25%–59.9%` risk):** Inconclusive gray zone flagged for radiologist review.
  * **Tuberculosis (`≥60%` risk):** High-confidence detection prioritized for immediate clinical response.
* **Native C++ Image Decoding:** Bypasses slow Python/PIL operations using `tf.io.decode_image` directly in C++ RAM.
* **Compiled Graph Execution:** Uses `@tf.function(reduce_retracing=True)` to convert inference into a static C++ execution graph.
* **Warmup Engine:** Executes dummy tensor graph compilation on startup to eliminate first-request latency.
* **Thread-Safe CPU Parallelism:** Limits `intra/inter_op_parallelism` to prevent CPU lockups and thread thrashing.
* **Production Docker Stack:** Containerized with `python:3.10-slim` and served via Gunicorn (1 worker, 2 threads) on **Port 5001**.

---

## 📁 Repository Structure

```text
├── app.py                     # Main Flask application with fast C++ image pipeline
├── densenet_tb_model.keras    # Trained DenseNet121 model weights (~51.9 MB)
├── Dockerfile                 # Production Docker configuration (Gunicorn + Python 3.10)
├── requirements.txt           # Dependency requirements (pinned numpy < 2.0.0)
├── .gitignore                 # Git rules ignoring environment and build artifacts
└── README.md                  # Project documentation
