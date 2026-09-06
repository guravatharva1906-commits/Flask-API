import os
# 1. Suppress logs & OpenMP thrashing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# 2. Prevent Windows CPU thread contention (Limits thread lockups)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

app = Flask(__name__)
CORS(app)

# 3. Load Model
MODEL_PATH = 'densenet_tb_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# 4. Compile prediction step into static C++ execution graph
@tf.function(reduce_retracing=True)
def fast_predict(tensor):
    return model(tensor, training=False)

# 5. Native TensorFlow Image Pipeline (Bypasses slow PIL library)
def fast_preprocess(image_bytes):
    # Decode directly in C++ RAM
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    
    # Crop, resize, and scale
    cropped = tf.image.central_crop(img, central_fraction=0.85)
    resized = tf.image.resize(cropped, (224, 224))
    preprocessed = tf.keras.applications.densenet.preprocess_input(resized)
    
    return tf.expand_dims(preprocessed, axis=0)

# Warmup compiled graph on startup
print("⚡ Warming up C++ graph execution engine...")
_ = fast_predict(tf.zeros((1, 224, 224, 3)))
print("✅ Server active on http://localhost:5000")

MEDICAL_SAFETY_THRESHOLD = 0.35

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Preprocess directly from bytes
        image_bytes = file.read()
        processed_tensor = fast_preprocess(image_bytes)
        
        # Execute compiled fast graph
        predictions = fast_predict(processed_tensor).numpy()
        
        prob_normal = float(predictions[0][0])
        prob_tb = float(predictions[0][1])
        
        is_tb = prob_tb >= MEDICAL_SAFETY_THRESHOLD
        label = 'Tuberculosis' if is_tb else 'Normal'
        primary_confidence = prob_tb if is_tb else prob_normal

        return jsonify({
            'success': True,
            'prediction': label,
            'is_tb_detected': is_tb,
            'confidence': round(primary_confidence * 100, 2),
            'probabilities': {
                'normal': round(prob_normal * 100, 2),
                'tuberculosis': round(prob_tb * 100, 2)
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)