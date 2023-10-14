# SMILES Molecular Property Prediction


## 💡 Overview

This section designed to predict properties of molecules based on their SMILES (Simplified Molecular Input Line Entry System) representation. The SMILES string is a notation allowing a user to represent a chemical structure in a way that can be used by the computer.

## Dependencies

- `pandas`: For data manipulation and analysis.
- `torch`: PyTorch, a deep learning framework.
- `sklearn`: For machine learning model evaluation metrics.

## How to Use ❓

### 1. Data Preprocessing 

Ensure your data is in the correct format. The model expects a CSV file with a column named "SMILES" that contains the SMILES strings of the molecules. Optionally, if you have labels for training/testing, ensure they are in a column named 'Label'.

```python
import pandas as pd

data = pd.read_csv("your_data.csv")
```
### 2. Feature Extraction
Convert the SMILES strings to a format suitable for machine learning using feature extraction. This might involve converting the SMILES string to a molecular graph, or some form of vectorization. Ensure you have a function smiles_to_fp to perform this action.

```python
data['features'] = data['SMILES'].apply(smiles_to_fp)
```
### 3. Model Prediction
Load the pre-trained model and use it to make predictions.

```python
import torch

model = YourModel()
model.load_state_dict(torch.load("path_to_your_model.pth"))
model.eval()

predictions = model.predict(data['features'])
```
### 4. Evaluation
Evaluate the model predictions using various metrics like accuracy, precision, recall, and F1 score. Ensure you have actual labels to compare the predictions against.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

actual_labels = ...
accuracy = accuracy_score(actual_labels, predictions)
precision = precision_score(actual_labels, predictions)
recall = recall_score(actual_labels, predictions)
f1 = f1_score(actual_labels, predictions)
```
