from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates 
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
import pickle
import os
from lightgbm import LGBMClassifier

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Loadding the data
data_test_sampled = pd.read_csv('data/data_test_sampled.csv', sep=',', index_col=[0], encoding='utf-8')
#data_test_sampled.set_index('SK_ID_CURR', inplace=True)

# Load the model from disk
model = pickle.load(open('lgbmt_best_model.pkl', 'rb'))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, client_id: int = Form(...)):
    """
    This function predicts a customer's creditworthiness on the basis of their identifier
    """
    # Check if the client_id exists
    if client_id not in data_test_sampled.index:
        error_message = f"Client ID {client_id} not found in database."
        return templates.TemplateResponse("index.html", {"request": request, "error_message": error_message})

    client_data = data_test_sampled.loc[[client_id]]

    # Making a prediction
    probability = model.predict_proba(client_data)[:, 1]  # Probability that the customer is class 1
    threshold = 0.133
    predicted_class = (probability > threshold).astype(int)  # Apply the threshold

    # Determine the message to return
    if predicted_class == 1:
        message = "<strong style='color:red;'>request refused</strong>"
    else:
        message = "<strong style='color:green;'>request accepted</strong>"

    # Return results with message.
    return templates.TemplateResponse("index.html", {
        "request": request,
        "client_id": client_id,
        "probability": round(probability[0], 3),
        "score": int(predicted_class[0]),
        "message": message
    })
class PredictionResponse(BaseModel):
    client_id: int
    probability: float
    score: int
    message: str

@app.get("/predict2")
async def predict2(client_id: int):
    """
    This function predicts a customer's creditworthiness based on their identifier.
    """
    # Check if the client_id exists
    if client_id not in data_test_sampled.index:
        raise HTTPException(status_code=404, detail=f"Client ID {client_id} not found in database.")

    client_data = data_test_sampled.loc[[client_id]]

    # Making a prediction
    probability = model.predict_proba(client_data)[:, 1]  # Probability that the customer is class 1
    threshold = 0.133
    predicted_class = (probability > threshold).astype(int)  # Apply the threshold

    # Determine the message to return
    if predicted_class == 1:
        message = "request refused"
    else:
        message = "request accepted"

    # Return results as JSON
    response = PredictionResponse(
        client_id=client_id,
        probability=round(probability[0], 3),
        score=int(predicted_class[0]),
        message=message
    )

    return response

if __name__ == '__main__':
    uvicorn.run(app)
