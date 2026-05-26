# app.py

from flask import Flask, request, jsonify, send_from_directory
import base64
import io
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from mnist_nn import NeuralNetwork

app = Flask(__name__)

model = NeuralNetwork.load("mnist_model.pkl")


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


def center_by_mass(arr):
    """Shift digit so center of mass is at image center — no scipy needed."""
    rows, cols = arr.shape
    total = arr.sum()
    if total < 1e-6:
        return arr  # blank canvas, skip

    # Compute center of mass manually
    ys = np.arange(rows).reshape(-1, 1)
    xs = np.arange(cols).reshape(1, -1)
    cy = float((ys * arr).sum() / total)
    cx = float((xs * arr).sum() / total)

    shift_y = int(round(rows / 2.0 - cy))
    shift_x = int(round(cols / 2.0 - cx))

    # Apply shift via numpy roll then zero out wrapped edges
    arr = np.roll(arr, shift_y, axis=0)
    if shift_y > 0:
        arr[:shift_y, :] = 0
    elif shift_y < 0:
        arr[shift_y:, :] = 0

    arr = np.roll(arr, shift_x, axis=1)
    if shift_x > 0:
        arr[:, :shift_x] = 0
    elif shift_x < 0:
        arr[:, shift_x:] = 0

    return arr


def preprocess(image_data_url):
    image_bytes = base64.b64decode(image_data_url.split(",")[1])

    # Open as RGBA, composite onto black bg to handle transparency
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    img = Image.alpha_composite(bg, img).convert("L")

    # Slight blur before downsampling
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # Resize digit into 20x20 bounding box, then pad to 28x28
    img = img.resize((20, 20), Image.LANCZOS)
    padded = Image.new("L", (28, 28), 0)
    padded.paste(img, (4, 4))

    arr = np.array(padded).astype(np.float32) / 255.0
    arr = center_by_mass(arr)
    arr = np.clip(arr, 0.0, 1.0)

    return arr


@app.route("/predict", methods=["POST"])
def predict():
    try:
        arr = preprocess(request.json["image"])
        probs = model.predict_proba(arr.reshape(1, 784))[0]
        prediction = int(np.argmax(probs))
        confidence = float(np.max(probs))

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probs.tolist(),
            "image28": arr.tolist()
        })
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
