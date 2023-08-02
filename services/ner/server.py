import copy
import datetime
import json
import subprocess
from logging import getLogger
from pathlib import Path
from typing import Optional, List

import pandas as pd
import uvicorn
from deeppavlov import build_model, deep_download
from deeppavlov.core.commands.utils import parse_config
from fastapi import FastAPI, File, UploadFile
from filelock import FileLock, Timeout
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware

from initial_setup import initial_setup
from main import evaluate, ner_config, metrics_filename, LOCKFILE, LOG_PATH

logger = getLogger(__file__)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

entity_detection_config = parse_config("entity_detection_rured.json")
entity_detection_lower_config = parse_config("entity_detection_collection3_lower.json")

deep_download("entity_detection_rured.json")
deep_download("entity_detection_collection3_lower.json")
initial_setup()

entity_detection = build_model(entity_detection_config, download=False)
entity_detection_lower = build_model(entity_detection_lower_config, download=False)


class Payload(BaseModel):
    x: List[str]


@app.post("/model")
async def model(payload: Payload):
    substr_b, offsets_b, pos_b, tags_b, sent_offsets_b, sent_b, probas_b = entity_detection(payload.x)
    substr_lw_b, offsets_lw_b, pos_lw_b, tags_lw_b, sent_offsets_lw_b, sent_lw_b, probas_lw_b = \
        entity_detection_lower(payload.x)

    substr_bt, offsets_bt, pos_bt, tags_bt, probas_bt = [], [], [], [], []
    for substr_l, offsets_l, pos_l, tags_l, probas_l, substr_lw_l, offsets_lw_l, pos_lw_l, tags_lw_l, probas_lw_l in \
            zip(substr_b, offsets_b, pos_b, tags_b, probas_b, substr_lw_b, offsets_lw_b, pos_lw_b, tags_lw_b,
                probas_lw_b):
        substr_t = copy.deepcopy(substr_l)
        offsets_t = copy.deepcopy(offsets_l)
        pos_t = copy.deepcopy(pos_l)
        tags_t = copy.deepcopy(tags_l)
        probas_t = copy.deepcopy(probas_l)
        for substr_lw, offsets_lw, pos_lw, tag_lw, probas_lw in \
                zip(substr_lw_l, offsets_lw_l, pos_lw_l, tags_lw_l, probas_lw_l):
            found = False
            for offsets in offsets_l:
                if offsets[0] <= offsets_lw[0] <= offsets[1] or offsets[0] <= offsets_lw[1] <= offsets[1]:
                    found = True
                    break
            if not found:
                substr_t.append(substr_lw)
                offsets_t.append(offsets_lw)
                pos_t.append(pos_lw)
                tags_t.append(tag_lw)
                probas_t.append(probas_lw)
        substr_bt.append(substr_t)
        offsets_bt.append(offsets_t)
        pos_bt.append(pos_t)
        tags_bt.append(tags_t)
        probas_bt.append(probas_t)

    res = {"entity_substr": substr_bt,
           "entity_offsets": offsets_bt,
           "entity_positions": pos_bt,
           "tags": tags_bt,
           "sentences_offsets": sent_offsets_b,
           "sentences": sent_b,
           "probas": probas_bt}
    return res


@app.get('/last_train_metric')
async def get_metric():
    last_metrics = {"success": False, "detail": "There is no metrics file. Call /evaluate to create"}
    if Path(metrics_filename).exists():
        df = pd.read_csv(metrics_filename)
        last_metrics = df.iloc[-1].to_dict()
        logger.warning(f"last_metrics {last_metrics}")

        last_metrics = {"success": True, "data": {"time": str(last_metrics["time"]),
                                                  "old_metric": float(last_metrics["old_metric"]),
                                                  "new_metric": float(last_metrics["new_metric"]),
                                                  "update_model": bool(last_metrics["update_model"])}}
    return last_metrics


@app.post("/train")
async def model_training(fl: Optional[UploadFile] = File(None)):
    data_path = "''"
    logger.info('Trying to start training')
    if fl:
        total_data = json.loads(await fl.read())
        if isinstance(total_data, list):
            train_data = total_data[:int(len(total_data) * 0.9)]
            test_data = total_data[int(len(total_data) * 0.9):]
        elif isinstance(total_data, dict) and "train" in total_data and "test" in total_data:
            train_data = total_data["train"]
            test_data = total_data["test"]
        else:
            raise HTTPException(status_code=400, detail="Train data should be either list with examples or dict with"
                                                        "'train' and 'test' keys")
        logger.info(f"train data {len(train_data)} test data {len(test_data)}")
        data_path = "/tmp/train_filename.json"
        with open(data_path, 'w', encoding="utf8") as out:
            json.dump({"train": train_data, "valid": test_data, "test": test_data},
                      out, indent=2, ensure_ascii=False)
    try:
        with FileLock(LOCKFILE, timeout=1):
            logfile = LOG_PATH / f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
            subprocess.Popen(['/bin/bash', '-c', f'python main.py {data_path}> {logfile} 2>&1'])
    except Timeout:
        logger.error("Can't start training since process is already running.")
        return {"success": False, "message": "Предыдущее обучение не завершено."}

    return {"success": True, "message": "Обучение инициировано"}


@app.get('/status')
async def proba():
    """Returns status of training process.
    Update functions use filelock to prevent starting multiple training processes simultaneously. In the end training
    function removes lock file. To check training process status this function checks if lockfile exists and
    either it acquired or not.
    """
    if LOCKFILE.exists():
        try:
            with FileLock(LOCKFILE, timeout=0.01):
                message = 'failed'
        except Timeout:
            message = 'running'
    else:
        message = 'finished sucessfully'
    return {'success': True, 'message': message}


@app.post("/evaluate")
async def model_testing(fl: Optional[UploadFile] = File(None)):
    if fl:
        test_data = json.loads(await fl.read())
        new_filename = "/tmp/test_filename.json"
        with open(new_filename, 'w', encoding="utf8") as out:
            if isinstance(test_data, list):
                json.dump({"train": [], "valid": [], "test": test_data},
                          out, indent=2, ensure_ascii=False)
            elif isinstance(test_data, dict) and "test" in test_data:
                json.dump({"train": [], "valid": [], "test": test_data["test"]},
                          out, indent=2, ensure_ascii=False)

        ner_config["dataset_reader"] = {
            "class_name": "sq_reader",
            "data_path": new_filename
        }
    cur_ner_f1, _ = evaluate(ner_config, False)

    return {"metrics": cur_ner_f1}


uvicorn.run(app, host='0.0.0.0', port=8000)
