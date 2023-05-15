import os
import warnings
import sys

# model
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from urllib.parse import urlparse

import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# comment types
all_types = ['อื่นๆ', 'ถนน', 'ทางเท้า', 'แสงสว่าง', 'ความปลอดภัย', 'ความสะอาด', 'น้ำท่วม', 'กีดขวาง',
             'ท่อระบายน้ำ', 'จราจร', 'สะพาน', 'สายไฟ', 'เสียงรบกวน', 'คลอง', 'ต้นไม้', 'ร้องเรียน', 'ป้าย',
             'สัตว์จรจัด', 'สอบถาม', 'PM2.5', 'เสนอแนะ', 'คนจรจัด', 'การเดินทาง', 'ห้องน้ำ', 'ป้ายจราจร']
toi = {t:idx for idx, t in enumerate(all_types)}

def make_feature_label(data_df: pd.DataFrame):
    X = []
    y = []
    for _, row in data_df[['type', 'hour_spend']].iterrows():
        type_vec = [0] * 25
        for t in row['type'].strip('{}').split(','):
            if t == '':
                type_vec[0] = 1
            else:
                type_vec[toi[t]] = 1
        X.append(type_vec)
        y.append(row['hour_spend'])
    X = np.array(X)
    y = np.array(y)
    return X, y

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**(1/2.0)
    score = r2_score(y_test, y_pred)

    return rmse, mse, score

if __name__ == "__main__":
    np.random.seed(2020)

    clean_df = pd.read_csv('data/cleaned_bangkok_traffy.csv', parse_dates=["timestamp"])
    mask = (clean_df.timestamp.dt.year <= 2022) & (clean_df.timestamp.dt.month < 7)
    after_df = clean_df[~mask]

    X, y = make_feature_label(after_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

    criterion = sys.argv[1] if len(sys.argv) > 1 else 'squared_error'
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else None
    min_samples_leaf = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    n_estimators = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    regr = RandomForestRegressor(n_estimators=n_estimators, 
                                 criterion=criterion,
                                 min_samples_leaf=min_samples_leaf,
                                 max_depth=max_depth,
                                 n_jobs=4)
    regr.fit(X_train, y_train)

    score = regr.score(X_train, y_train)
    print("""RandomForest regressor model (criterion={}, max_depth={}, 
            min_samples_leaf={} n_estimators={})""".format(criterion, max_depth, min_samples_leaf, n_estimators))
    print("  training R2:", score)

    (rmse, mse, r2) = evaluate_regression(regr, X_test, y_test)
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mse)
    print("  R2: %s" % r2)

    mlflow.log_param("criterion", criterion)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("n_estimators", n_estimators)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mse", mse)

    
    predictions = regr.predict(X_train)
    signature = infer_signature(X_train, predictions)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(
            regr, "model", registered_model_name="ElasticnetWineModel", signature=signature
        )
    else:
        mlflow.sklearn.log_model(regr, "model", signature=signature)