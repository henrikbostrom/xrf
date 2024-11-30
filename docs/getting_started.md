# Getting Started

## Installation

From [PyPI](https://pypi.org/project/xrf/)

```bash
pip install xrf
```

## Quickstart

### Classification forests

Let us start by importing the tic-tac-toe dataset from [openml.org](www.openml.org).

```python
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder

dataset = fetch_openml(name="tic-tac-toe", parser="auto")

y = dataset.target.values

X = OneHotEncoder().fit_transform(dataset.data.values).toarray()
```

Let us split the dataset into a training and a test set.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75)
```

Let us now fit an explainable random forest classifier; we can use the same parameters as for standard random forest classifiers as implemented in `scikit-learn`.

```python
from xrf import XRandomForestClassifier

rfx = XRandomForestClassifier(n_jobs=-1)
rfx.fit(X_train, y_train)
```

We get the predictions in the usual way, using either `predict` or `predict_proba`, here resulting in exactly the same output as the standard random forest classifiers in `scikit-learn`. 

```python
rfx.predict_proba(X_test)
```

```numpy
array([[0.05, 0.95],
       [0.56, 0.44],
       [0.4 , 0.6 ],
       ...,
       [0.21, 0.79],
       [0.17, 0.83],
       [0.59, 0.41]])
```

We may now limit the number of examples involved in a prediction, e.g., to at most 5.

```python
rfx.predict_proba(X_test, k=5)
```

```numpy
array([[0.        , 1.        ],
       [0.85416634, 0.14583366],
       [0.34500622, 0.65499378],
       ...,
       [0.27464175, 0.72535825],
       [0.12693503, 0.87306497],
       [1.        , 0.        ]])
```

Let us also obtain the example attributions, by setting `return_examples` and `return_weights` to `True`.

```python
predictions, examples, weights = rfx.predict_proba(X_test, k=5, 
                                                   return_examples=True, 
                                                   return_weights=True)
```

Let us also take a look at the example attributions; `examples` will contain the indexes of the training objects involved in each prediction, while `weights` will contain the corresponding weights.

```python
examples
```

```numpy
array([[ 26, 131,  40, 193, 169],
       [ 48, 121,  52, 164,   6],
       [203, 176, 213, 110,  99],
       ...,
       [ 52, 167, 194, 175,  53],
       [104,  71,  20,  35, 122],
       [ 33,  47, 188, 228, 120]])
```

```python
weights
```

```numpy
array([[0.23050922, 0.21026052, 0.19812573, 0.18882078, 0.17228375],
       [0.24554293, 0.20930998, 0.20651394, 0.19279949, 0.14583366],
       [0.2935989 , 0.25979051, 0.21957101, 0.12543522, 0.10160437],
       ...,
       [0.27464175, 0.23320384, 0.19853987, 0.15467345, 0.13894108],
       [0.32220957, 0.21056097, 0.20287181, 0.13742261, 0.12693503],
       [0.26857466, 0.20863132, 0.20008477, 0.18560888, 0.13710037]])
```

### Regression forests

Let us import the Miami housing dataset from [openml.org](www.openml.org).

```python
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder

dataset = fetch_openml(name="miami_housing", parser="auto")

y = dataset.target.values
X = dataset.data.values
```

Let us split the dataset into a training and a test set.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75)
```

Let us generate and apply an explainable random forest regressor without constraining the number of training examples involved in the predictions.

```python
from xrf import XRandomForestRegressor

rfx = XRandomForestRegressor(n_jobs=-1)
rfx.fit(X_train, y_train)
rfx.predict(X_test)
```

```numpy
array([492859., 193170., 260507., ..., 330824., 416856., 241969.])
```

We may now limit the number of examples involved in a prediction, e.g., to at most 5.

```python
rfx.predict(X_test, k=5)
```

```numpy
array([541411.11111111, 196994.81865285, 210900.81300813, ...,
       340516.66666667, 389410.25641026, 241550.27422303])
```

The example attributions are obtained by setting `return_examples` and `return_weights` to `True`.

```python
predictions, examples, weights = rfx.predict(X_test, k=5,
                                             return_examples=True,
                                             return_weights=True)
```

We may check that the predictions are the same as the weighted targets of the training examples.

```python
import numpy as np

weighted_predictions = np.sum([weights[i]*y_train[examples[i]] 
                               for i in range(len(weights))], axis=1)

np.allclose(predictions, weighted_predictions)
```

```python
True
```

You are welcome to download and try out `xrf`; you may find the following notebook helpful:

[xrf.ipynb](https://github.com/henrikbostrom/xrf/blob/main/docs/xrf.ipynb)