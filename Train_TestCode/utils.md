# 🚀 `utils.py` Comprehensive Guide 📚

This document serves as a detailed explanation of the functionalities present in the `utils.py` file, accompanied by fun emojis for a more intuitive understanding.

## 📦 Imports

The following packages and modules are imported:

- Core Python modules: `logging`.
- Data science packages: `numpy` (as `np`), `pandas` (as `pd`), and various functions from `sklearn`.
- Deep learning framework: `torch` (as `t`).
- Custom model-related functions from the `model` module.

## 🌍 Global Variables

There are several global variables defined:

- `X`: List containing strings "X1" to "X1024".
- `Y`: String "potency".
- `label`: String "SMILES".
- `xcols`: A list generated based on twice the length of `X`.
- `combine_modes`: A list containing strings ['subtract', 'add', 'cos', 'concat'].
- Some `pandas` display options are set.

## 📐 Functions

### 1️⃣ `metrics(y, y_pred)`

Compute various metrics like accuracy, precision, recall, F1 score, ROC AUC score, and confusion matrix based on true and predicted labels.

### 2️⃣ `evaluation(y_matrix, y_test, thresh=0.5, noneg=False)`

Evaluate the performance of a model on a given test set. It returns a dataframe containing various evaluation metrics for the proteins.

### 3️⃣ `evaluation_multilabel(y_matrix,y_test, thresh=0.5)`

Evaluate the performance for multi-label problems. Returns accuracy, precision, recall, F1 score, ROC AUC score, average precision, and a confusion matrix.

### 4️⃣ `predict_cos(model,x,n_support=100,iter=100, support = None, random_seed=None)`

Predict the cosine similarity for a given model and input data.

### 5️⃣ `predict_general(model, x, n_support=100,iter=500, support = None,random_seed=None)`

Generically predict for a given model and input data.

### 6️⃣ `trainCycle(params, save=True)`

Train a model with specified parameters. Data is loaded, split, and the model is then trained. The trained model can be saved optionally.

### 7️⃣ `transferTrainCycle(model,params, save=True)`

Transfer learning cycle: Use a pre-trained model and train it on a new dataset with given parameters.

### 8️⃣ `testCycle(model, params, saveName=None, test_raw=None,thresh=0.5,seed=None,noneg=False)`

Test the performance of a model on a test dataset and save the results if needed.

## 🌟 Conclusion

This file, `utils.py`, provides essential functionalities for metrics calculation, model evaluation, predictions, and model training cycles. Each function has been designed with a specific purpose to assist in the overall machine learning pipeline.

