# 🧠 Handwritten Digit Recognition using CNN (PyTorch)

A deep learning project that recognizes handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

This project was built using PyTorch and demonstrates:

- CNN architecture
- Image preprocessing
- Model training
- Accuracy evaluation
- Custom image prediction
- GPU support with CUDA

---

# 🚀 Features

✅ Trains a CNN on the MNIST dataset  
✅ Evaluates test accuracy  
✅ Predicts custom handwritten digit images  
✅ Uses Dropout for regularization  
✅ Supports GPU acceleration  
✅ Beginner-friendly deep learning project

---

# 🛠 Technologies Used

- Python
- PyTorch
- TorchVision
- Matplotlib
- PIL (Python Imaging Library)

---

# 🧠 CNN Architecture

The model contains:

- Convolution Layer 1
- Convolution Layer 2
- Max Pooling
- Dropout Layer
- Fully Connected Layers

The network learns important image features and classifies digits from 0–9.

---

# 📂 Dataset

This project uses the famous MNIST handwritten digit dataset.

Dataset contains:

- 60,000 training images
- 10,000 testing images
- Grayscale 28×28 images

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/MNIST-Digit-Recognizer-CNN.git
cd MNIST-Digit-Recognizer-CNN
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Project

```bash
python main.py
```

---

# 📊 Model Performance

The CNN achieves high accuracy on the MNIST test dataset.

Typical Accuracy:

```text
~98% Test Accuracy
```

---

# 🔍 Custom Image Prediction

You can test your own handwritten digit image.

Steps:

1. Write a digit on paper
2. Take a photo
3. Crop the digit
4. Save image
5. Update image path inside `predict_digit()`

Example:

```python
predict_digit("digit.png")
```

---

# 📸 Project Output

The model displays:

- Predicted digit
- Image visualization
- Training loss
- Test accuracy

---

# 📚 Learning Outcomes

This project helped in understanding:

- Convolutional Neural Networks (CNNs)
- Deep Learning workflow
- Forward propagation
- Backpropagation
- Optimizers
- Image preprocessing
- PyTorch fundamentals

---

# 🔮 Future Improvements

- Add GUI interface
- Deploy as web app
- Train on custom datasets
- Add real-time camera prediction
- Save and load trained models

---

# 👨‍💻 Author

Mithun Ram

Beginner AI/ML Developer passionate about Deep Learning and Artificial Intelligence.

---
