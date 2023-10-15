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
## Visualization
### ✍️ Predicted vs Experimental pIC50 Chart
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

plt.figure(figsize=(8, 8))

sns.scatterplot(x=y_train, y=predictions_train, marker='o', color='blue', alpha=1, label='Train', s=100)

sns.scatterplot(x=y_test, y=predictions_test, marker='^', color='yellow', edgecolor='red', linewidth=0.5, alpha=1, label='Test', s=100)

plt.plot([y.min(), y.max()], 
         [y.min(), y.max()], 
         color='black', linestyle='--')

plt.xlabel("Experimental pIC50", fontsize=14, labelpad=10)
plt.ylabel("Predicted pIC50", fontsize=14, labelpad=10)
plt.title("Predicted vs Experimental pIC50 values", fontsize=16, pad=10)

plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

```
## ✍️ Residual Chart【Residual=Observed−Predicted】
```python
import matplotlib.pyplot as plt
import seaborn as sns

residuals_test = y_test - predictions_test
residuals_train = y_train - predictions_train

plt.figure(figsize=(8, 6))

sns.scatterplot(x=predictions_train, y=residuals_train, marker='o', color='blue', alpha=1, label='Train', s=100)

sns.scatterplot(x=predictions_test, y=residuals_test, marker='^', color='yellow',edgecolor='red', linewidth=0.5, alpha=1, label='Test', s=100)

plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel("Predicted pIC50", fontsize=14, labelpad=10)
plt.ylabel("Residuals", fontsize=14, labelpad=10)
plt.title("Residuals of Predicted pIC50 values", fontsize=16, pad=10)

plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
```
## ✍️ Applicability Domain Anaysis
```python
for i, name in enumerate(unique_names):
    mask_train = names_train == name
    num_total_samples = np.sum(mask_train)
    num_filtered_samples = np.sum(mask_train & (y_train < 5.0))  # Example condition
    plt.scatter(X_pca_train[mask_train, 0], X_pca_train[mask_train, 1], color=colors[i], marker=markers[i], 
                label=f'Train {name} (total: {num_total_samples}, filtered: {num_filtered_samples})', s=73)

for i, name in enumerate(test_in_ad_names):
    mask_test = (names_test == name) & is_inside_ad
    num_test_samples = np.sum(mask_test)
    plt.scatter(X_pca_test[mask_test, 0], X_pca_test[mask_test, 1], color=colors[unique_names.index(name)], 
                marker=markers[unique_names.index(name)], edgecolors='k', 
                label=f'Test In AD {name} (total: {num_test_samples})', s=73)

plt.xlabel(f'PC1: {pca.explained_variance_ratio_[0]:.2%} explained variance')
plt.ylabel(f'PC2: {pca.explained_variance_ratio_[1]:.2%} explained variance')
plt.title('Applicability Domain Analysis')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.annotate('Note: Samples were filtered based on a pIC50 threshold of 5.0.', 
             (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=9)

plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()




