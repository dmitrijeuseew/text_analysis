import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from process import process


class Item(BaseModel):
    data: dict

app = FastAPI()


@app.get("/")
async def root():
    return {"status":"ready"}


@app.post("/get_analysis/")
async def get_analysis(data: Item):
    result = process(data.data)
    return result


uvicorn.run(app, host='0.0.0.0', port=8003)
