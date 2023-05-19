# Local imports
import datetime

# Third party imports
from pydantic import BaseModel, Field

from ms import app
from ms.functions import get_model_response


model_name = "Traffyfondue's task hour spend prediction"
version = "v0.2.0"


# Input for data validation
class Input(BaseModel):
    types: list[str]
    organization: list[str]
    comment: str
    latitude: float
    longitude: float
    
# Ouput for data validation
class Output(BaseModel):
    prediction: float


@app.get('/info')
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": model_name,
        "version": version
    }


@app.get('/health')
async def service_health():
    """Return service health"""
    return {
        "ok"
    }


@app.post('/predict', response_model=Output)
async def model_predict(input: Input):
    """Predict with input"""
    response = get_model_response(input)
    return response