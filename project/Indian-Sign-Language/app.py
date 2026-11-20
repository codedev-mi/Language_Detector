from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import base64
import os
import threading
from collections import deque

app = Flask(__name__)

# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "sign_language_mobilenet.h5"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "label_encoder_mobilenet.pkl"
)

# =========================================================
# LOAD MODEL AND LABEL ENCODER
# =========================================================

print("Loading sign language model...")

model = tf.keras.models.load_model(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

print("Model loaded successfully!")

# =========================================================
# MEDIAPIPE CONFIGURATION
# =========================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================================================
# GLOBAL VARIABLES
# =========================================================

is_running = False

sentence = ""
current_letter = ""
prev_char = ""

prediction_buffer = deque(maxlen=15)

# Locks prevent concurrent requests from modifying shared state
state_lock = threading.Lock()
inference_lock = threading.Lock()

# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")


# =========================================================
# START DETECTION
# =========================================================

@app.route("/start", methods=["POST"])
def start_detection():

    global is_running

    with state_lock:

        is_running = True

        prediction_buffer.clear()

        return jsonify({
            "status": "success",
            "message": "Detection started"
        })


# =========================================================
# STOP DETECTION
# =========================================================

@app.route("/stop", methods=["POST"])
def stop_detection():

    global is_running
    global sentence
    global current_letter
    global prev_char

    with state_lock:

        is_running = False

        prediction_buffer.clear()

        current_letter = ""
        prev_char = ""

        return jsonify({
            "status": "success",
            "message": "Detection stopped",
            "sentence": sentence
        })


# =========================================================
# CLEAR SENTENCE
# =========================================================

@app.route("/clear", methods=["POST"])
def clear_sentence():

    global sentence
    global current_letter
    global prev_char

    with state_lock:

        sentence = ""
        current_letter = ""
        prev_char = ""

        prediction_buffer.clear()

        return jsonify({
            "status": "success",
            "message": "Sentence cleared"
        })


# =========================================================
# GET CURRENT SENTENCE
# =========================================================

@app.route("/get_sentence")
def get_sentence():

    with state_lock:

        return jsonify({
            "sentence": sentence,
            "current": current_letter,
            "running": is_running
        })


# =========================================================
# PREDICTION API
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    global sentence
    global current_letter
    global prev_char

    with state_lock:

        if not is_running:

            return jsonify({
                "status": "stopped",
                "current": current_letter,
                "sentence": sentence
            })

    try:

        data = request.get_json()

        if not data or "image" not in data:

            return jsonify({
                "status": "error",
                "message": "No image received"
            }), 400

        image_data = data["image"]

        # Remove Base64 header
        if "," in image_data:

            image_data = image_data.split(",")[1]

        # Decode image
        image_bytes = base64.b64decode(image_data)

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "status": "error",
                "message": "Invalid image"
            }), 400

        # =================================================
        # MEDIAPIPE HAND DETECTION
        # =================================================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        with inference_lock:

            results = hands.process(frame_rgb)

            if not results.multi_hand_landmarks:

                with state_lock:

                    current_letter = ""

                    return jsonify({
                        "status": "no_hand",
                        "current": "",
                        "sentence": sentence,
                        "confidence": 0
                    })

            hand_landmarks = results.multi_hand_landmarks[0]

            height, width, _ = frame.shape

            x_coordinates = [
                int(landmark.x * width)
                for landmark in hand_landmarks.landmark
            ]

            y_coordinates = [
                int(landmark.y * height)
                for landmark in hand_landmarks.landmark
            ]

            # Bounding box with padding
            x_min = max(min(x_coordinates) - 20, 0)
            x_max = min(max(x_coordinates) + 20, width)

            y_min = max(min(y_coordinates) - 20, 0)
            y_max = min(max(y_coordinates) + 20, height)

            hand_image = frame[
                y_min:y_max,
                x_min:x_max
            ]

            if hand_image.size == 0:

                return jsonify({
                    "status": "no_hand",
                    "current": "",
                    "sentence": sentence,
                    "confidence": 0
                })

            # =================================================
            # PREPROCESSING
            # =================================================

            roi = cv2.resize(
                hand_image,
                (128, 128)
            )

            roi = roi.astype("float32") / 255.0

            roi = np.expand_dims(
                roi,
                axis=0
            )

            # =================================================
            # MODEL PREDICTION
            # =================================================

            predictions = model.predict(
                roi,
                verbose=0
            )

            class_id = np.argmax(predictions[0])

            confidence = float(
                predictions[0][class_id]
            )

            label = le.inverse_transform(
                [class_id]
            )[0]

        # =================================================
        # PREDICTION STABILIZATION
        # =================================================

        with state_lock:

            if confidence > 0.7:

                prediction_buffer.append(label)

            stable_prediction = ""

            if len(prediction_buffer) == prediction_buffer.maxlen:

                stable_prediction = max(
                    set(prediction_buffer),
                    key=prediction_buffer.count
                )

                current_letter = stable_prediction

                # Avoid repeatedly adding the same gesture
                if stable_prediction != prev_char:

                    if stable_prediction == "SPACE":

                        sentence += " "

                    elif stable_prediction == "DEL":

                        sentence = sentence[:-1]

                    else:

                        sentence += stable_prediction

                    prev_char = stable_prediction

            return jsonify({
                "status": "success",
                "prediction": label,
                "stable_prediction": stable_prediction,
                "current": current_letter,
                "confidence": confidence,
                "sentence": sentence
            })

    except Exception as error:

        print("Prediction error:", error)

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500


# =========================================================
# APPLICATION START
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
