import pandas as pd
import numpy as np

# evaluation
from sklearn.metrics import mean_squared_error, r2_score

mapping = {
    'type': [],
    'organization': []
}

with open('request_type.txt', 'r', encoding='utf-8') as f:
    for line in f:
        mapping['type'].append(line.strip())

with open('organizations.txt', 'r', encoding='utf-8') as f:
    for line in f:
        mapping['organization'].append(line.strip())


def make_binary_feature(data_series: pd.Series, mapping_key: str):
    assert mapping_key in mapping
    features = []
    for _, val in data_series.items():
        val_vec = np.zeros(len(mapping[mapping_key]))
        for x in val:
            try:
                idx = mapping[mapping_key].index(x)
                val_vec[idx] += 1
            except ValueError:
                # print('invalid type')
                pass
        features.append(val_vec)
    return pd.DataFrame(features, columns=mapping[mapping_key])

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**(1/2.0)
    score = r2_score(y_test, y_pred)
    return rmse, mse, score