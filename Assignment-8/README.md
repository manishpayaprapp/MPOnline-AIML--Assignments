# Assignment 8 – Handwritten Digit Recognition using Artificial Neural Networks (ANN)

### **Name:** Manish Satish Payaprapp
### **Reg.No:** 23BCY10046
### **Application.No:** IN26009666

## Objective
Build an Artificial Neural Network (ANN) using TensorFlow/Keras to classify handwritten digits (0–9) from the MNIST dataset, simulating an automated postal code recognition system for a postal service organization.

## Dataset Link
MNIST Handwritten Digits Dataset — Kaggle:
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

> Note: The dataset is **not included in this repository**. Download it from the Kaggle link above (`mnist_train.csv` and/or `mnist_test.csv`) and place the CSV in the project root as `mnist_data.csv` before running the notebook. This project was developed and evaluated using the 10,000-row MNIST test-set CSV (785 columns: 1 label + 784 pixel values per row), split 80/20 for training/testing within the notebook.

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (`train_test_split`, evaluation metrics)
- tensorflow / keras (`Sequential`, `Dense`, `to_categorical`)

## Methodology
1. **Data Understanding** – Loaded the CSV with Pandas, displayed the first five records, identified the 784 pixel columns as input features and the label column as the target, reviewed dataset dimensions/info, and displayed a sample handwritten digit image with Matplotlib.
2. **Data Preprocessing** – Checked for missing values, separated features (X) and target (y), normalized pixel values from the 0–255 range to 0–1, split the data 80% training / 20% testing, and one-hot encoded the labels into a 10-column categorical format.
3. **Model Development** – Built and trained a Keras `Sequential` ANN (architecture below), compiled with the Adam optimizer and categorical crossentropy loss, trained for 10 epochs, and generated predictions on the test set.
4. **Model Evaluation** – Evaluated test accuracy, generated a confusion matrix and classification report, and plotted Accuracy vs Epoch and Loss vs Epoch curves.

## Model Architecture
| Layer | Type | Units | Activation |
|---|---|---|---|
| Input | — | 784 | — |
| Hidden Layer 1 | Dense | 128 | ReLU |
| Hidden Layer 2 | Dense | 64 | ReLU |
| Output Layer | Dense | 10 | Softmax |

**Compilation:** Optimizer = Adam, Loss = Categorical Crossentropy, Metric = Accuracy
**Training:** 10 epochs, batch size 32

## Results
| Metric | Score |
|---|---|
| Test Accuracy | 0.9545 |
| Test Loss | 0.1644 |

**Per-class performance highlights:** precision ranged from 0.87 (digit 9) to 1.00 (digit 1); recall ranged from 0.92 (digits 4 and 7) to 0.98 (digits 0, 6, and 9). Full classification report and confusion matrix are in the notebook.

The sample digit image, confusion matrix, and Accuracy/Loss vs Epoch curves are generated inside the notebook (`Assignment-8.ipynb`) and saved as `sample_digit.png`, `confusion_matrix.png`, and `accuracy_loss_vs_epoch.png`.

**Key observations:**
1. The ANN reaches ~95.5% test accuracy after just 10 epochs, showing a simple fully-connected architecture can learn digit classification directly from normalized pixel values.
2. Training curves show a growing train/validation gap from around epoch 6–7 onward — training accuracy climbs to ~99.8% while validation accuracy plateaus near 95–96%, and validation loss even ticks back up slightly in later epochs — a clear sign of **overfitting** as training continues.
3. The model's weaker spots are concentrated in specific digit pairs (digit 9 has the lowest precision, digit 4 the lowest recall), consistent with how visually similar these digits can be depending on handwriting style.
4. This run used a 10,000-image MNIST test-set CSV split 80/20 rather than the full 60,000-image official training set; a larger training set or regularization techniques (dropout, early stopping) would likely reduce the overfitting seen and improve generalization further.

## Conclusion
The ANN with two hidden layers (128 and 64 neurons, ReLU activation) and a softmax output layer reached about 95.5% test accuracy after 10 epochs, confirming it can learn meaningful digit-classification patterns directly from normalized pixel intensities without manual feature engineering. Training also revealed mild overfitting in later epochs, a reminder that more epochs isn't always better without regularization.

Hidden layers matter because they let the network learn increasingly abstract, non-linear representations of the input — early layers can capture simple patterns like edges or strokes, while deeper layers combine these into higher-level shapes corresponding to digit structures; without them, the model would be limited to a purely linear decision boundary, insufficient for this visually complex task.

One advantage of Deep Learning over traditional ML is that it automatically learns relevant features directly from raw data, removing the need for the manual feature engineering that classical approaches (like KNN or Decision Trees) typically require. One limitation of ANNs is that they need substantial labeled data and compute to perform well, are prone to overfitting without regularization (as observed here), and act largely as a "black box," making their internal decision-making harder to interpret than simpler models.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
jupyter nbconvert --to notebook --execute --inplace Assignment-8.ipynb
```
