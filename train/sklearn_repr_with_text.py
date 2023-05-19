import os
import warnings
import sys
from utils import *

# data preprocessing
import pythainlp
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer

# model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor

from urllib.parse import urlparse


import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    randseed = 54
    
    criterion = sys.argv[1] if len(sys.argv) > 1 else 'squared_error'
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else None
    max_features= float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    min_samples_leaf = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    n_estimators = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    min_df = int(sys.argv[6]) if len(sys.argv) > 6 else 3

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
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("min_df", min_df)
        mlflow.log_param("seed", randseed)

        numeric_transformer = Pipeline(steps = [('MinMaxScaler', MinMaxScaler())])
        text_vectorizer = Pipeline(steps=[('TextVectorizer', CountVectorizer(min_df=min_df, tokenizer=pythainlp.word_tokenize, analyzer='word'))])

        preprocessor = ColumnTransformer(transformers = [('nums', numeric_transformer, ['latitude', 'longitude']),
                                                 ('text',text_vectorizer, 'comment')])
        # model
        model = RandomForestRegressor(n_estimators=n_estimators, 
                                    criterion=criterion,
                                    max_features=max_features,
                                    min_samples_leaf=min_samples_leaf,
                                    max_depth=max_depth,
                                    random_state=randseed,
                                    n_jobs=-1)
        # model pipeline
        modelPipeline = Pipeline(steps = [('preprocessor', preprocessor),
                                        ('model', model)])
        modelPipeline.fit(X_train, y_train)
        
        score = modelPipeline.score(X_train, y_train)
        print("""RandomForest regressor model (criterion={}, max_depth={}, max_features={}
                min_samples_leaf={} n_estimators={})""".format(criterion, max_depth, max_features, min_samples_leaf, n_estimators))
        print("  training LogR2:", score)

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
                modelPipeline, "model", registered_model_name="RandomForestTraffyModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(modelPipeline, "rf_reg_model", signature=signature)