import os
import warnings
import sys
from utils import *

# data preprocessing
import pythainlp
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

# model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet

from urllib.parse import urlparse


import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    randseed = 54
    
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 1
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    min_df = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    train = pd.read_json('data/train.json').reset_index(drop=True)
    test = pd.read_json('data/test.json').reset_index(drop=True)
    type_features_train = make_binary_feature(train['type'], 'type')
    org_features_train = make_binary_feature(train['organization'], 'organization')
    
    type_features_test = make_binary_feature(test['type'], 'type')
    org_features_test = make_binary_feature(test['organization'], 'organization')
    
    X_train = pd.concat((type_features_train, org_features_train, train[['comment', 'latitude', 'longitude']]), axis=1)
    y_train = train['hour_spend'].values


    X_test = pd.concat((type_features_test, org_features_test, test[['comment', 'latitude', 'longitude']]), axis=1)
    y_test = test['hour_spend'].values
    
    with mlflow.start_run():
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("min_df", min_df)
        mlflow.log_param("seed", randseed)

        numeric_transformer = Pipeline(steps = [('StandardScaler', StandardScaler())])
        text_vectorizer = Pipeline(steps=[('TextVectorizer', CountVectorizer(min_df=min_df, tokenizer=pythainlp.word_tokenize, analyzer='word'))])

        preprocessor = ColumnTransformer(transformers = [('nums', numeric_transformer, ['latitude', 'longitude']),
                                                 ('text',text_vectorizer, 'comment')])
        # model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=randseed)
        # model pipeline
        modelPipeline = Pipeline(steps = [('preprocessor', preprocessor),
                                        ('model', model)])
        modelPipeline.fit(X_train, y_train)
        
        score = modelPipeline.score(X_train, y_train)
        print("""Elasticnet model (alpha={}, l1_ratio={})""".format(alpha, l1_ratio))
        print("  training R2:", score)

        # evaluate model in original scale
        (rmse, mse, r2) = evaluate_regression(modelPipeline, X_test, y_test)
        print("###### Evaluate in original scale ######")
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mse)
        print("  R2: %s" % r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mse", mse)

        predictions = modelPipeline.predict(X_train)
        signature = infer_signature(X_train, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                modelPipeline, "model", registered_model_name="ElasticNetTraffyModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(modelPipeline, "elasticnet_model_with_text", signature=signature)