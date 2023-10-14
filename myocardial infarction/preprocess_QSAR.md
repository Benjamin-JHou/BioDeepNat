## QSAR Analysis for Model Validation

Before validating the model with SMILES data, a QSAR (Quantitative Structure-Activity Relationship) analysis was performed. 

### 📚 Import Necessary Libraries 
```python
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
```
### 🧹 Data Preprocessing 
```python
df = pd.read_csv('activity_pic50.csv') # Before doing this make sure to convert IC50 to pIC50
df = df.dropna(subset=['SMILES'])
df = df[df['SMILES'].apply(lambda x: isinstance(x, str))]
```
### 👆 Computing Fingerprints 
```python
def compute_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  
    return pd.Series(list(fp))

df_fingerprints = df['SMILES'].apply(compute_fingerprint)
df = pd.concat([df, df_fingerprints], axis=1)
```
### 🚀 Model Training 
```python
X = df[df.columns[-1024:]]
y = df['pIC50']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
X = df[df.columns[-1024:]]
y = df['pIC50']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
```
### 📊 Feature Importance 
```python
importances = model.feature_importances_
plt.barh(range(1, 1025), importances)  
plt.xlabel("Feature Importance")
plt.ylabel("Bit position")
plt.show()
```
### 🔄 Model Retraining with Selected Features 
```python
selected_features = X.columns[importances > np.mean(importances)]
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

model.fit(X_train_selected, y_train)
```
### 🎯 Model Evaluation 
```python
# For Train Data
predictions_train = model.predict(X_train_selected)
mse_train = mean_squared_error(y_train, predictions_train)
r2_train = r2_score(y_train, predictions_train)

# Display for Train Data
print("\\nPerformance on TRAIN data:")
print(f'Mean Squared Error: {mse_train}')
print(f'R2 Score: {r2_train}')
print(f'Mean Absolute Error: {mean_absolute_error(y_train, predictions_train)}')
print(f'Root Mean Squared Error: {np.sqrt(mse_train)}')

# For Test Data
predictions_test = model.predict(X_test_selected)
mse_test = mean_squared_error(y_test, predictions_test)
r2_test = r2_score(y_test, predictions_test)
mae_test = mean_absolute_error(y_test, predictions_test)
rmse_test = np.sqrt(mse_test)

# Display for Test Data
print("\\nPerformance on TEST data:")
print(f'Mean Squared Error: {mse_test}')
print(f'R2 Score: {r2_test}')
print(f'Mean Absolute Error: {mae_test}')
print(f'Root Mean Squared Error: {rmse_test}')
```
