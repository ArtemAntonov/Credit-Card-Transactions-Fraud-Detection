from fastapi import Body, FastAPI
import numpy as np
import uvicorn
import dill
import lzma
from pydantic import BaseModel
import pandas as pd
from typing import Annotated

pred_examples={
                "fraud": {
                    "summary": "Fraudulent transaction",
                    "description": "An example of fraudulent transaction",
                    "value": {
                        "trans_date_trans_time": "2020-12-16 02:02:52",
                        "cc_num": 2714019737356678,
                        "merchant": "fraud_Hackett-Lueilwitz",
                        "category": "grocery_pos",
                        "amt": 291.1,
                        "first": "Christina",
                        "last": "Moore",
                        "gender": "F",
                        "street": "542 Finley Ports Apt. 396",
                        "city": "West Finley",
                        "state": "PA",
                        "zip": 15377,
                        "lat": 39.9914,
                        "long": -80.4408,
                        "city_pop": 724,
                        "job": "Theme park manager",
                        "dob": "1995-08-30",
                        "trans_num": "32a55d346cceb51cfe68a451e31986d3",
                        "unix_time": "1387159372",
                        "merch_lat": 39.835289,
                        "merch_long": -81.103024
                    }
                },
                "non_fraud": {
                    "summary": "Non-fraudulent transaction",
                    "description": "An example of non-fraudulent transaction",
                    "value": {
                        "trans_date_trans_time": "2020-12-31 23:24:05",
                        "cc_num": 30118423745458,
                        "merchant": "fraud_Little Ltd",
                        "category": "kids_pets",
                        "amt": 38.92,
                        "first": "Jared",
                        "last": "Velazquez",
                        "gender": "M",
                        "street": "01479 Murray Circle",
                        "city": "Matawan",
                        "state": "NJ",
                        "zip": 7747,
                        "lat": 40.4109,
                        "long": -74.238,
                        "city_pop": 30770,
                        "job": "Drilling engineer",
                        "dob": "1993-04-29",
                        "trans_num": "ca86d7ec05ba0591d515003a8a475acf",
                        "unix_time": "1388532245",
                        "merch_lat": 39.826996,
                        "merch_long": -74.709078
                    }
                }
            }

class Item(BaseModel):
    trans_date_trans_time: str
    cc_num: int
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: int
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    trans_num: str
    unix_time: str
    merch_lat: float
    merch_long: float

class Pred(BaseModel):
    fraud: int

app = FastAPI()

with lzma.open('prediction_pipeline.xz', 'rb') as file:
    pred_pipe = dill.load(file)

@app.get("/")
async def root():
    return {
        "Name": "Credit Card Transactions Fraud Detection",
        "Description": "Predicts if the transaction is fraudulent based on given information."
    }

@app.post("/predict/", response_model=Pred)
def predict(data: Annotated[Item, Body(openapi_examples=pred_examples)]): # type: ignore
    df = pd.DataFrame([data.model_dump()])    
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    
    prediction = pred_pipe.predict(df)
    
    return Pred(fraud=prediction[0])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)