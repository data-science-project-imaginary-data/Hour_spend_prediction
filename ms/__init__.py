from fastapi import FastAPI
import pickle

# Initialize FastAPI app
app = FastAPI()

# Load model
with open('mlruns\\0\\24f9cba9f6a049088653d0c13be97e42\\artifacts\model\model.pkl', 'rb') as f:
    model = pickle.load(f)