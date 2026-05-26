# MNIST Digit Recognizer — Neural Network from Scratch

A handwritten digit recognizer built without any deep learning frameworks. The neural network is implemented from scratch in NumPy and served via a Flask web app with a live drawing canvas.

---

## Demo

Draw a digit on the canvas, hit **Predict**, and the model returns its prediction along with a confidence score and per-class probability breakdown. A 28×28 preview shows exactly what the model sees.

---

## Architecture

```
Input (784)  →  Dense (256, ReLU)  →  Dense (128, ReLU)  →  Output (10, Softmax)
```

- **Optimizer:** SGD with momentum (lr=0.01, momentum=0.9)
- **Weight init:** He initialization
- **Loss:** Cross-entropy
- **Test accuracy:** ~97% on MNIST test set

---

## Project Structure

```
mnist_app/
├── mnist_nn.py       # Neural network, training loop, MNIST loader
├── app.py            # Flask backend + image preprocessing
├── index.html        # Drawing canvas frontend
├── mnist_model.pkl   # Saved weights (generated after training)
└── mnist_data/       # Cached MNIST data (generated after training)
```

---

## Setup

```bash
# Install dependencies
pip install flask numpy pillow scipy tensorflow

# Train the model (downloads MNIST automatically)
python mnist_nn.py

# Start the app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

> **Mac users:** If you hit an SSL error during training, run:
> `/Applications/Python\ 3.11/Install\ Certificates.command`

---

## How It Works

**Training (`mnist_nn.py`)**
- Implements forward pass, backpropagation, and SGD with momentum entirely in NumPy
- Downloads and caches real MNIST data (60,000 train / 10,000 test)
- Trains for 20 epochs with mini-batch size 128

**Preprocessing (`app.py`)**
- Canvas PNG is composited onto a black background and converted to grayscale
- Resized to 20×20 with Gaussian smoothing, then padded to 28×28
- Digit is centered by center of mass to match MNIST's preprocessing pipeline

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Neural network math |
| `flask` | Web server |
| `pillow` | Image preprocessing |
| `scipy` | Center-of-mass digit centering |
| `tensorflow` | MNIST data download (optional fallback) |

---

## Author

Tarun Mahesh
