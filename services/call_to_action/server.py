import argparse
import json
import os
from typing import List
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from deeppavlov import build_model


app = FastAPI()

batch_size = 60

class Payload(BaseModel):
    texts: List[str]


agency_cls_model = build_model("agency_cls.json", download=True)

@app.post("/model")
async def model(payload: Payload):
    texts = payload.texts
    texts_with_probas = []
    try:
        num_batches = len(texts) // batch_size + int(len(texts) % batch_size > 0)

        for i in range(num_batches):
            cur_texts = texts[i*batch_size:(i+1)*batch_size]
            y_pred, probas = agency_cls_model(cur_texts)
            text_with_probas = {}
            for text, cls, proba in zip(cur_texts, y_pred, probas):
                text_with_probas["text"] = text
                text_with_probas["agency_class"] = cls
                text_with_probas["agency_proba"] = float(round(proba[2], 2))
                texts_with_probas.append(text_with_probas)
    except Exception as e:
        print(f"error: {e}")
    return texts_with_probas


uvicorn.run(app, host='0.0.0.0', port=8004)