# Assignment 9 — Image Classification using CNN (Cats vs Dogs)

### **Name:** Manish Satish Payaprapp
### **Reg.No:** 23BCY10046
### **Application.No:** IN26009666

## Objective
Build a Convolutional Neural Network (CNN) to classify images of pets into **Cat** and **Dog**
categories, as requested by an animal welfare organization looking to automate this process.

## Dataset Link
[Dog and Cat Classification Dataset — Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

The dataset is **not included in this repository** (per the assignment instructions) — download it
directly from Kaggle using the link above, or via the Kaggle API as shown in the notebook.

## Libraries Used
- TensorFlow / Keras — model building, training, `ImageDataGenerator`
- scikit-learn — evaluation metrics (accuracy, precision, recall, F1, confusion matrix)
- NumPy, Pillow — array/image handling
- Matplotlib — plotting sample images, training curves, confusion matrix
- Kaggle API — dataset download

## Methodology
1. **Data Understanding** — Downloaded the dataset via the Kaggle API, inspected the folder
   structure, displayed sample images from each class, and recorded class counts and image
   dimensions.
2. **Data Preprocessing** — Filtered out any corrupt/unreadable image files, split the cleaned
   data 80/20 into train/test folders, then used Keras `ImageDataGenerator` to resize all images
   to 128×128 and rescale pixel values to the 0–1 range.
3. **Model Development** — Built a CNN with three Conv2D + MaxPooling2D blocks
   (32 → 64 → 128 filters), followed by a Flatten layer, a 128-unit Dense ReLU layer, and a
   single sigmoid output neuron for binary classification. Compiled with the Adam optimizer and
   binary crossentropy loss, trained for 10 epochs.
4. **Model Evaluation** — Evaluated on the held-out test set with accuracy, precision, recall,
   and F1-score; visualized results with a confusion matrix and accuracy/loss-vs-epoch curves.

## CNN Architecture
```
Input (128, 128, 3)
Conv2D(32, 3x3, ReLU)      -> MaxPooling2D(2x2)
Conv2D(64, 3x3, ReLU)      -> MaxPooling2D(2x2)
Conv2D(128, 3x3, ReLU)     -> MaxPooling2D(2x2)
Flatten
Dense(128, ReLU)
Dense(1, Sigmoid)

Optimizer: Adam
Loss:      Binary Crossentropy
Metric:    Accuracy
Epochs:    10
```

## Results
> **Note:** This repository ships the notebook fully written and ready to run, but **not
> pre-executed** — running it requires a Kaggle API token and either a local GPU or a free
> Google Colab GPU runtime (training 10 epochs on the full dataset on CPU would be very slow).
> Fill in the actual numbers below after running `Assignment-9.ipynb` end-to-end:

| Metric | Value |
|---|---|
| Test Accuracy | *[fill in]* |
| Precision | *[fill in]* |
| Recall | *[fill in]* |
| F1-Score | *[fill in]* |

Result plots (`sample_images.png`, `confusion_matrix.png`, `accuracy_loss_vs_epoch.png`) will be
generated in the working directory when you run the notebook.

## Conclusion
See the Task 5 markdown cell in `Assignment-9.ipynb` for the full write-up (key findings, the role
of convolution/pooling layers, a CNN-vs-ANN advantage, and a CNN limitation) — update the bracketed
metric values there once you have real results from your run.

## How to Run
1. Open `Assignment-9.ipynb` in **Google Colab**.
2. Runtime → Change runtime type → **GPU**.
3. Get a Kaggle API token: Kaggle → Account → *Create New API Token* (downloads `kaggle.json`).
4. Run all cells top to bottom — the first code cell will prompt you to upload `kaggle.json` and
   will download the dataset automatically.
