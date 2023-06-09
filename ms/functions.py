import os
import pandas as pd
from train.utils import make_binary_feature
from dotenv import load_dotenv
import urllib

import mlflow

load_dotenv()


logged_model = os.getenv('MODEL_PATH')

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

def predict(data_df):
    # model prediction
    type_feature = make_binary_feature(data_df['types'], 'type')
    org_feature = make_binary_feature(data_df['organization'], 'organization')
    Input = pd.concat((type_feature, org_feature, data_df[['comment', 'latitude', 'longitude']]), axis=1)

    prediction = loaded_model.predict(Input)
    return prediction


def get_model_response(input):
    X = pd.json_normalize(input.__dict__)
    prediction = predict(X)
    return {
        'prediction': prediction,
    }