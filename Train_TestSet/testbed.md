# 📁 Testbed for Model Training and Evaluation 

## 📚 Import Necessary Libraries 

```python
import logging
from preprocess import preprocess_directory, addSupport
from model import loadModel
from utils import (
    trainCycle,
    testCycle,
    transferTrainCycle,
    tfTestCycle,
    evaluation,
    evaluation_multilabel,
)
import pandas as pd
```
      
## 🔧 Parameter Setting 

```python
X = ['X%d' % i for i in range(1, 1025)]  # fingerprint columns
Y = "potency"  # classification column
label = "SMILES"  # raw fingerprint column
exceptions = []  # names to not include in training
path = 'BaseSet'  # directory containing csv's
exname = ''  # extra name to add to saved results
tfpath = 'TransferSet'  # path of transfer data
exname2 = ''  # extra name for transfer data

nname = path + exname
nnname = tfpath + exname2
params = dict(
    name=nname,
    tfname=nnname,
    headshape=[2048],
    bodyshape=[],
    dr=0.5,
    combine_mode='cos',
    lr=0.0001,
    nsupport=100,
    niter=100,
)

modelname = "%s_%s_%.2f_%s_%s" % (
    params['name'],
    params['combine_mode'],
    params['dr'],
    '-'.join([str(i) for i in params['headshape']]),
    '-'.join([str(i) for i in params['bodyshape']]),
)

tfmodelname = "%s_%s_%.2f_%s_%s_tf%s" % (
    params['name'],
    params['combine_mode'],
    params['dr'],
    '-'.join([str(i) for i in params['headshape']]),
    '-'.join([str(i) for i in params['bodyshape']]),
    params['tfname'],
)

```
## 🔄 Creating Data Pairs 

```python
create_glob_set(
    path, X, Y, label, 7000, 7000, nnpairs=0, exceptions=exceptions, exname=exname, testsize=0.1, maxrows=None
)

```
## 🌀 Training Cycle 

```python
model, losses, _ = trainCycle(params)
losses.to_csv("%s_losses.csv" % modelname)

```
## ✅ Testing Cycle 

```python
model, losses, _ = trainCycle(params)
losses.to_csv("%s_losses.csv" % modelname)

```
## ➡️ Transfer Learning 

```python
create_glob_set(
    tfpath, X, Y, label, 7000, 7000, nnpairs=0, exceptions=exceptions, exname=exname2, testsize=0.1, maxrows=None
)

model = loadModel("model_%s.pt" % (modelname))
tmodel, losses, _ = transferTrainCycle(model, params)

```
# ✅ Transfer Testing 

```python
tmodel = loadModel("model_%s.pt" % (tfmodelname))
y_proba, y_matrix = tfTestCycle(tmodel, params, saveName=tfmodelname, thresh=0.5, seed=777)


```
## ♻️ Test Cycle with Different Support (Fewshot) 


```python
model = loadModel("model_%s.pt" % (tfmodelname))
test_raw = pd.DataFrame()
support_set = dict()

```
## 📊 Add support from directory 

```python
support_set, test_raw = addSupport(Y, 'FewshotSet', support_set, test_raw, testPerc=0.1)
model.support_pos = support_set

y_test = test_raw.loc[:, 'class']
y_proba, y_matrix = testCycle(model, params, saveName=None, test_raw=test_raw, thresh=0.5)

```
## 🧲 Cosine Similarity 

```python
from siamese_general_fit import predict_cos
import numpy as np

model = loadModel("model_%s.pt" % (modelname))
test_raw = pd.read_csv(nname + "_sim_test.csv")
y_proba = predict_cos(model, test_raw[X].values.astype(np.float64), n_support=100, iter=100, random_seed=777)

y_test = test_raw.loc[:, 'class']

y_matrix = y_proba.groupby(axis=1, level=0).max()
y_matrix['LABEL'] = y_test
y_matrix[label] = test_raw[label]
y_matrix = y_matrix.reset_index(drop=True)

res = neo_evaluation(y_matrix, y_test, thresh=0.6, noneg=True)
print(res)

multi_ytest = pd.read_csv("%sml_sim_test.csv" % nname)
acc, pre, rec, f1, roc, ap, cf = evaluation_multilabel(y_matrix, multi_ytest, thresh=0.6)
print("acc, pre, rec, f1, roc, ap")
print(acc, pre, rec, f1, roc, ap)


```
## 🏷️ Multi-Label Learning 

```python
path = 'multi_label_data'
exname = ''
nname = path + exname
nnname = tfpath + exname2
params = dict(
    name=nname,
    tfname=nnname,
    headshape=[2048],
    bodyshape=[],
    dr=0.5,
    combine_mode='cos',
    lr=0.0001,
    nsupport=100,
    niter=100,
)

modelname = "%s_%s_%.2f_%s_%s" % (
    params['name'],
    params['combine_mode'],
    params['dr'],
    '-'.join([str(i) for i in params['headshape']]),
    '-'.join([str(i) for i in params['bodyshape']]),
)

tfmodelname = "%s_%s_%.2f_%s_%s_tf%s" % (
    params['name'],
    params['combine_mode'],
    params['dr'],
    '-'.join([str(i) for i in params['headshape']]),
    '-'.join([str(i) for i in params['bodyshape']]),
    params['tfname'],
)

model, losses, _ = trainCycle(params)
losses.to_csv("%s_losses.csv" % modelname)

```
# 📔 Single label results 

```python
model = loadModel("model_%s.pt" % (modelname))
y_proba, y_matrix = testCycle(model, params, saveName=modelname, thresh=0.5, seed=777, noneg=True)

multi_ytest = pd.read_csv("%sml_sim_test.csv" % nname)
acc, pre, rec, f1, roc, ap, cf = evaluation_multilabel(y_matrix, multi_ytest)
print("acc, pre, rec, f1, roc, ap")
print(acc, pre, rec, f1, roc, ap)
