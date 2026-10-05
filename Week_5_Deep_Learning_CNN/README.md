# Week 5 - Deep Learning Application in Data Science

## Project
**Handwritten Digit Classification using a Convolutional Neural Network (CNN)**

## Objective
The objective of this project is to understand the complete deep-learning workflow by building a CNN that classifies handwritten digits from 0 to 9.

## Dataset
The project uses the **scikit-learn Digits dataset**. It contains 8x8 grayscale images of handwritten digits with 10 target classes (0-9). The dataset is small enough to run on a normal CPU while still demonstrating convolutional neural-network concepts.

## Why CNN?
CNNs are designed to learn spatial patterns in images. Convolutional layers detect local features, pooling reduces spatial size, and fully connected layers perform the final classification.

## Architecture
- Input: 1 x 8 x 8 grayscale image
- Conv2D: 1 -> 32 filters, 3x3 kernel
- ReLU
- MaxPool2D
- Conv2D: 32 -> 64 filters, 3x3 kernel
- ReLU
- MaxPool2D
- Flatten
- Dense: 256 -> 128
- ReLU
- Dropout: 0.30
- Output: 10 classes

## Training Configuration
- Framework: PyTorch
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Epochs: 15
- Loss: CrossEntropyLoss
- Weight decay: 1e-4
- Train/validation/test split: 68% / 12% / 20% approximately

## Results
The included report contains the actual run results, loss/accuracy curves, confusion matrix, sample predictions, and critical analysis.

**Test accuracy from the included run: 97.22%**

## Folder Structure
```text
Week_5_Deep_Learning_CNN/
├── src/
│   └── train.py
├── results/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── architecture_diagram.png
│   ├── dataset_samples.png
│   ├── sample_predictions.png
│   └── digit_cnn.pt
├── docs/
│   └── Week_5_Deep_Learning_Report.docx
├── README.md
└── requirements.txt
```

## How to Run

### 1. Create a virtual environment
```bash
python -m venv venv
```

### 2. Activate it
Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the model
```bash
python src/train.py
```

The dataset is loaded directly through scikit-learn, so no manual dataset download is required.

## Key Challenges
1. Avoiding overfitting on a relatively small dataset.
2. Choosing a CNN architecture that is expressive without being unnecessarily large.
3. Keeping training practical on CPU hardware.
4. Interpreting class-level errors using the confusion matrix.

## Learning Outcomes
This project demonstrates dataset preparation, normalization, train/validation/test splitting, CNN design, training, validation, evaluation, confusion-matrix analysis, and model saving.

## Author
**Pragati Maurya**
