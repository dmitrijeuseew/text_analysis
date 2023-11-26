import json
from logging import getLogger
from typing import Optional, List
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from deeppavlov import build_model


logger = getLogger(__file__)
app = FastAPI()

batch_size = 60

class Payload(BaseModel):
    texts: List[str]


topic_cls = build_model("topic_cls_chatgpt_22.json", download=True)
subtopic_cls = build_model("topic_cls_chatgpt_120.json", download=True)
topics2_health = build_model("topics2_health_education.json", download=True)
topics3_health = build_model("topics3_health_education.json", download=True)

topics2_h_list = []
with open("/data/models/classifiers/topics2_h_e/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics2_h_list.append(line_split[0])

topics3_h_list = []
with open("/data/models/classifiers/topics3_h_e/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics3_h_list.append(line_split[0])

with open("/data/models/classifiers/topics_by_categories.json", 'r') as inp:
    topics_by_categories = json.load(inp)

file_subtopics = "/data/downloads/topic_subtopics.json"
subtopics_dict = json.load(open(file_subtopics))

bashkir_letters = "ҙҡәҫңғ"
alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя1234567890.,!?;:'%()-«» " + '"' + '/'
alphabet_full = alphabet + "abcdefghijklmnopqrstuvwxyz"


def preprocess_text(text):
    found = False
    for letter in bashkir_letters:
        if letter in text:
            found = True
            break
    if not found:
        text = text.replace("\n", " ").replace("   ", " ").replace("  ", " ")
        text = ''.join([symb for symb in text if symb.lower() in alphabet_full])
        text_split = text.split()
        text_split = [tok for tok in text_split if "http" not in tok and "#" not in tok and ".ru/" not in tok]
        if len(text_split) > 7:
            text_split = text_split[:150]
            text = " ".join(text_split)
            return text, found
    return text, found


@app.post("/model")
async def model(payload: Payload):
    texts = payload.texts

    processed_elements = []
    num_batches = len(texts) // batch_size + int(len(texts) % batch_size > 0)
    for i in range(num_batches):
        try:
            f_texts, nf_texts = [], []
            cur_texts = texts[i*batch_size:(i+1)*batch_size]
            for n, text in enumerate(cur_texts):
                if text is None:
                    text = ""
                f_text, found = preprocess_text(text)
                if found:
                    nf_texts.append([f_text, n])
                else:
                    f_texts.append([f_text, n])

            f_labels, nf_labels = [], []
            raw_texts = [element[0] for element in f_texts]
            if raw_texts:
                labels = topic_cls(raw_texts)
                labels_sp = subtopic_cls(raw_texts)
                labels2_h, probas2_h = topics2_health(raw_texts)
                labels3_h, probas3_h = topics3_health(raw_texts)

                for nl, (label, label2_h, proba2_h, label3_h, proba3_h, label_sp) in \
                        enumerate(zip(labels, labels2_h, probas2_h, labels3_h, probas3_h, labels_sp)):
                    label2_probas_h = zip(topics2_h_list, proba2_h)
                    label2_probas_h = sorted(label2_probas_h, key=lambda x: x[1], reverse=True)
                    if label2_probas_h[0][1] < 0.92 and label.lower() not in ["здравоохранение", "образование"]:
                        label2_h = "Не найдено"

                    if label.lower() == "здравоохранение" and label2_h not in topics_by_categories["topics2_h"]:
                        label2_h = "Не найдено"
                    if label.lower() == "образование" and label2_h not in topics_by_categories["topics2_e"]:
                        label2_h = "Не найдено"

                    label3_probas_h = zip(topics3_h_list, proba3_h)
                    label3_probas_h = sorted(label3_probas_h, key=lambda x: x[1], reverse=True)
                    if label2_h == "Не найдено":
                        label3_h = "Не найдено"

                    if label.lower() == "здравоохранение" and label3_h not in topics_by_categories["topics3_h"]:
                        label3_h = "Не найдено"
                    if label.lower() == "образование" and label3_h not in topics_by_categories["topics3_e"]:
                        label3_h = "Не найдено"

                    f_labels.append([label, label2_h, label3_h, label_sp, f_texts[nl][1]])

            for _, n in nf_texts:
                nf_labels.append(["Не найдено", "Не найдено", "Не найдено", ["Не найдено"], n])

            total_labels = f_labels + nf_labels
            total_labels = sorted(total_labels, key=lambda x: x[-1])
            cur_twl = [[text, element[0], element[1], element[2], element[3]] for text, element in zip(cur_texts, total_labels)]

            new_cur_twl = []
            for text, label, label2_h, label3_h, label_sp in cur_twl:
                if label2_h != "Не найдено":
                    label = "Здравоохранение"
                    label_sp = label2_h
                elif label == "Не найдено":
                    label_sp = "Не найдено"
                elif label_sp[0] == "Не найдено":
                    label_sp = "Не найдено"
                elif label != "Не найдено" and label_sp and label_sp[0] != "Не найдено":
                    subtopics = subtopics_dict.get(label, [])
                    fs = False
                    for subtopic in label_sp[:10]:
                        if subtopic in subtopics:
                            label_sp = subtopic
                            fs = True
                            break
                    if not fs:
                        label_sp = "Не найдено"
                else:
                    label = "Не найдено"
                    label_sp = "Не найдено"

                processed_elements.append({"text": text, "topic1": label, "topic2": label_sp, "topic3": label3_h})
        except Exception as e:
            print(f"error: {e}")
            cur_texts = texts[i*batch_size:(i+1)*batch_size]
            for text in cur_texts:
                processed_elements.append({"text": text,
                                           "topic1": "Не найдено",
                                           "topic2": "Не найдено",
                                           "topic3": "Не найдено"})
    return processed_elements


uvicorn.run(app, host='0.0.0.0', port=8002)
