# Changelog

## v0.1.0 (30/11/2024)

### Features

- The classes `XRandomForestClassifier` and `XRandomForestRegressor` implement random forest classifiers and regressors with example attribution, i.e., each prediction is associated with a weight distribution over the training examples. The examples used in forming a prediction can be limited by their number (`k`) or by their cumulative weight (`c`).
