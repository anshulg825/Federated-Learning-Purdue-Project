>>> from creme import compose
>>> from creme import datasets
>>> from creme import linear_model
>>> from creme import metrics
>>> from creme import preprocessing

X_y = datasets.Phishing()

model = compose.Pipeline(preprocessing.StandardScaler(),
                             linear_model.LogisticRegression())

metric = metrics.Accuracy()

for x, y in X_y:
    y_pred = model.predict_one(x)      # make a prediction
    metric = metric.update(y, y_pred)  # update the metric
    model = model.fit_one(x, y)        # make the model learn

print(metric)