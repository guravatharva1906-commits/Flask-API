import os
# 1. Suppress verbose TensorFlow logs & OpenMP thrashing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# 2. Prevent CPU thread contention on Windows/macOS/Servers
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

app = Flask(__name__)
CORS(app)

# 3. Load Trained DenseNet121 Model
MODEL_PATH = 'densenet_tb_model.keras'
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# 4. Compile prediction step into static C++ execution graph
@tf.function(reduce_retracing=True)
def fast_predict(tensor):
    return model(tensor, training=False)

# 5. Native TensorFlow C++ Image Pipeline (Bypasses slow PIL library)
def fast_preprocess(image_bytes):
    # Decode directly in C++ RAM
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    
    # Crop (0.85), resize (224x224), and apply DenseNet scaling
    cropped = tf.image.central_crop(img, central_fraction=0.85)
    resized = tf.image.resize(cropped, (224, 224))
    preprocessed = tf.keras.applications.densenet.preprocess_input(resized)
    
    return tf.expand_dims(preprocessed, axis=0)

# 6. Warmup compiled C++ graph on startup to prevent 1st request latency
print("⚡ Warming up C++ graph execution engine...")
_ = fast_predict(tf.zeros((1, 224, 224, 3)))
print("✅ Server active and model warmed up successfully!")

# 7. Clinical Threshold Boundaries (0.0 to 1.0 scale)
NORMAL_THRESHOLD = 0.25        # < 25% TB risk = Normal
TB_CONFIDENT_THRESHOLD = 0.60  # >= 60% TB risk = High-Risk Tuberculosis

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Flexible key check: Accepts either 'file' or 'image' form-data keys
    file_key = 'file' if 'file' in request.files else ('image' if 'image' in request.files else None)
    if not file_key:
        return jsonify({'error': 'No file uploaded under key "file" or "image"'}), 400
    
    file = request.files[file_key]
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Preprocess directly from bytes via native TF C++ RAM pipeline
        image_bytes = file.read()
        processed_tensor = fast_preprocess(image_bytes)
        
        # Execute compiled fast C++ graph
        predictions = fast_predict(processed_tensor).numpy()
        
        prob_normal = float(predictions[0][0])
        prob_tb = float(predictions[0][1])
        
        # 3-Tier Clinical Triage Logic
        if prob_tb < NORMAL_THRESHOLD:
            label = 'Normal'
            is_tb = False
            primary_confidence = prob_normal
            recommendation = 'No further immediate imaging required.'

        elif prob_tb >= TB_CONFIDENT_THRESHOLD:
            label = 'Tuberculosis'
            is_tb = True
            primary_confidence = prob_tb
            recommendation = 'Prioritize for immediate clinical confirmation and treatment.'

        else:
            # Gray Zone (25.0% to 59.9% TB Probability)
            label = 'Examination Required'
            is_tb = False
            primary_confidence = max(prob_normal, prob_tb)
            recommendation = 'Inconclusive scan: Flagged for manual radiologist review or secondary testing.'

        return jsonify({
            'success': True,
            'prediction': label,
            'is_tb_detected': is_tb,
            'confidence': round(primary_confidence * 100, 2),
            'probabilities': {
                'normal': round(prob_normal * 100, 2),
                'tuberculosis': round(prob_tb * 100, 2)
            },
            'recommendation': recommendation
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Port 5001 avoids macOS AirPlay Receiver conflict on Port 5000, 
    # leaving Port 3000 free for Express.js Gateway
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)