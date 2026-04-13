from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import cv2
import os

app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl", "rb"))

def prepare_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))   # adjust size based on your training
    img = img.flatten()               # flatten for ML model
    img = img / 255.0
    return np.array([img])

def predict_image(image_path):
    img = prepare_image(image_path)
    if img is None:
        return None, None, None

    result = model.predict(img)
    prediction = "Male" if result[0] == 1 else "Female"

    # Get probabilities if model supports it
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(img)[0]
        female_prob = proba[0] * 100
        male_prob = proba[1] * 100
    else:
        male_prob = 100.0 if result[0] == 1 else 0.0
        female_prob = 100.0 - male_prob

    return prediction, male_prob, female_prob

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    male_prob = None
    female_prob = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        path = os.path.join("static/uploads", file.filename)
        os.makedirs("static/uploads", exist_ok=True)
        file.save(path)
        image_path = "uploads/" + file.filename

        prediction, male_prob, female_prob = predict_image(path)
        if prediction is None:
            prediction = "Error"

    return render_template("index.html",
                           prediction=prediction,
                           male_prob=male_prob,
                           female_prob=female_prob,
                           image_path=image_path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Save uploaded file
    filepath = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    # Predict
    prediction, male_prob, female_prob = predict_image(filepath)

    if prediction is None:
        return jsonify({'error': 'Could not read image'})

    return jsonify({
        'prediction': prediction,
        'male_probability': f"{male_prob:.2f}%",
        'female_probability': f"{female_prob:.2f}%"
    })

if __name__ == "__main__":
    app.run(debug=True)