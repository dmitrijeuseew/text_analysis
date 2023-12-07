import argparse
import json
import os
import pickle
from logging import getLogger
from typing import Optional, List
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from deeppavlov import build_model


parser = argparse.ArgumentParser()
parser.add_argument("-n", action="store", dest="number")
parser.add_argument("-d", action="store", dest="device")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

batch_size = 40

topics2_health = build_model("topics2_health.json", download=True)
topics3_health = build_model("topics3_health.json", download=True)

topics2_h_list = []
with open("./data/models/classifiers/topics2_healthcare/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics2_h_list.append(line_split[0])

topics3_h_list = []
with open("./data/models/classifiers/topics3_healthcare/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics3_h_list.append(line_split[0])

with open("./data/models/classifiers/topics_by_categories.json", 'r') as inp:
    topics_by_categories = json.load(inp)

file_subtopics = "./data/downloads/topic_subtopics.json"
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


nf = 0
with open(f"texts_health/{args.number}.json", 'r') as inp:
    data = json.load(inp)

print("started")
processed_elements = []
num_batches = len(data) // batch_size + int(len(data) % batch_size > 0)
for i in range(num_batches):
    try:
        f_texts, nf_texts = [], []
        cur_samples = data[i*batch_size:(i+1)*batch_size]
        cur_texts = [smp["text"] for smp in cur_samples]
        cur_text_ids = [smp.get("id", 0) for smp in cur_samples]
        cur_probas1 = [smp.get("proba1", 0) for smp in cur_samples]
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
            labels2_h, probas2_h = topics2_health(raw_texts)
            labels3_h, probas3_h = topics3_health(raw_texts)

            for nl, (label2_h, proba2_h, label3_h, proba3_h) in \
                    enumerate(zip(labels2_h, probas2_h, labels3_h, probas3_h)):
                label2_probas_h = zip(topics2_h_list, proba2_h)
                label2_probas_h = sorted(label2_probas_h, key=lambda x: x[1], reverse=True)

                label3_probas_h = zip(topics3_h_list, proba3_h)
                label3_probas_h = sorted(label3_probas_h, key=lambda x: x[1], reverse=True)
                if label2_h == "Не найдено":
                    label3_h = "Не найдено"

                f_labels.append([(label2_h, label2_probas_h[0][1]), (label3_h, label3_probas_h[0][1]), f_texts[nl][1]])

        for _, n in nf_texts:
            nf_labels.append([("Не найдено", 1.0), ("Не найдено", 1.0), n])

        total_labels = f_labels + nf_labels
        total_labels = sorted(total_labels, key=lambda x: x[-1])
        cur_twl = [[text_id, text, proba1, element[0], element[1]]
                   for text_id, text, proba1, element in zip(cur_text_ids, cur_texts, cur_probas1, total_labels)]

        new_cur_twl = []
        for text_id, text, proba1, (label2_h, proba2), (label3_h, proba3) in cur_twl:
            if label3_h == "0":
                label3_h = "Не найдено"
            processed_elements.append({"id": text_id, "text": text, "topic1": "здравоохранение", "topic2": label2_h,
                                       "topic3": label3_h, "proba1": float(proba1), "proba2": float(proba2),
                                       "proba3": float(proba3)})
    except Exception as e:
        print(f"error: {e}")
        cur_samples = texts[i*batch_size:(i+1)*batch_size]
        cur_texts = [smp["text"] for smp in cur_samples]
        cur_text_ids = [smp.get("id", 0) for smp in cur_samples]
        cur_probas1 = [smp.get("proba1", 0) for smp in cur_samples]
        for text_id, text, proba1 in zip(cur_text_ids, cur_texts, cur_probas1):
            processed_elements.append({"text": text,
                                       "topic1": "здравоохранение",
                                       "topic2": "Не найдено",
                                       "topic3": "Не найдено",
                                       "proba1": proba1,
                                       "proba2": 1.0,
                                       "proba3": 1.0})
    if i % 100 == 0 and i > 0:
        with open(f"processed_health/{args.number}_{nf}.json", 'w', encoding="utf8") as out:
            json.dump(processed_elements, out, indent=2, ensure_ascii=False)
        processed_elements = []
        nf += 1
        print("iter", nf)

if processed_elements:
    with open(f"processed_health/{args.number}_{nf}.json", 'w', encoding="utf8") as out:
        json.dump(processed_elements, out, indent=2, ensure_ascii=False)
