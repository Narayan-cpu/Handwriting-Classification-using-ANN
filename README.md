# 📝 Handwriting Classification Using ANN with Keras & TensorFlow

A neural network model built with TensorFlow and Keras to classify handwritten digits (e.g., using the MNIST dataset). This project trains a simple ANN to recognize handwritten digits with high accuracy and provides scripts for training, evaluation, and prediction.

# 🚀 Project Overview

Handwritten digit classification is a classic machine learning problem. This project uses an Artificial Neural Network (ANN) implemented with Keras (TensorFlow backend) to classify 28×28 grayscale images of handwritten digits (0–9).

# 📌 Features

✔ Data preprocessing (normalization & reshaping)
✔ Train, validate & test model
✔ Save & load trained model
✔ Predict on new handwritten samples
✔ Easy to understand and extend

# 📦 Tech Stack

Python 3.x

TensorFlow

Keras

NumPy

Matplotlib (optional for visualization)

# 📁 Repository Structure
handwriting-classification-ann/
├── data/
│   └── custom_samples/             # Optional: your own handwriting images
├── models/
│   └── handwriting_ann.h5          # Saved trained model
├── src/
│   ├── train.py                    # Train the ANN model
│   ├── evaluate.py                 # Evaluate model on test data
│   ├── predict.py                  # Run predictions on new images
│   └── utils.py                    # Preprocessing & helper functions
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE

# 📥 Installation

Clone this repo

git clone https://github.com/Narayan-cpu/Handwriting-Classification-using-ANN.git

cd handwriting-classification-ann


Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt

# 🧠 Training the Model

Run the training script:

python src/train.py


The model will train on the MNIST dataset and save trained weights to models/handwriting_ann.h5.

# 📊 Evaluate Model

Evaluate performance on test data:

python src/evaluate.py


Sample output metrics will include:

Accuracy

Loss

# 🖋 Predict Handwriting

To run predictions on custom image samples:

python src/predict.py --image path/to/sample.png


Make sure input images are:
✔ Grayscale
✔ 28×28 pixels
✔ Black background with white digit

# 🔧 Customizing the Model

You can change:

Layers and neurons in the ANN

Activation functions

Learning rate and optimizer

Epochs and batch size

# 🧪 Example Results
Digit	Prediction
7	7
3	3
0	0
9	9

Sample predictions will be logged in the console and can optionally be plotted.

# 🙌 Contributing

Contributions are welcome! Feel free to:

Open issues

Add new features

Improve documentation

Share better models or visualizations

