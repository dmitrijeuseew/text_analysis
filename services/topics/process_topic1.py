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

batch_size = 60

topic_cls = build_model("topic_cls_chatgpt_22.json", download=True)

topics1_list = []
with open("./data/models/topic_cls_chatgpt_base/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics1_list.append(line_split[0])

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
with open(f"texts_november/{args.number}.pickle", 'rb') as inp:
    samples = pickle.load(inp)

print("started")
processed_elements = []
num_batches = len(samples) // batch_size + int(len(samples) % batch_size > 0)
for i in range(num_batches):
    try:
        f_texts, nf_texts = [], []
        cur_samples = samples[i*batch_size:(i+1)*batch_size]
        cur_texts = [sample["text"] for sample in cur_samples]
        cur_text_ids = [sample["id"] for sample in cur_samples]
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
            labels, probas = topic_cls(raw_texts)

            for nl, (label, proba) in enumerate(zip(labels, probas)):
                label1_probas = zip(topics1_list, proba)
                label1_probas = sorted(label1_probas, key=lambda x: x[1], reverse=True)

                f_labels.append([(label, label1_probas[0][1]), f_texts[nl][1]])

        for _, n in nf_texts:
            nf_labels.append([("Не найдено", 1.0), n])

        total_labels = f_labels + nf_labels
        total_labels = sorted(total_labels, key=lambda x: x[-1])
        cur_twl = [[text_id, text, element[0]]
                   for text_id, text, element in zip(cur_text_ids, cur_texts, total_labels)]

        new_cur_twl = []
        for text_id, text, (label, proba1) in cur_twl:
            if label.lower() == "здравоохранение":
                processed_elements.append({"id": text_id, "text": text, "topic1": label, "proba1": float(round(proba1, 2))})
    except Exception as e:
        print(f"error: {e}")
        cur_samples = samples[i*batch_size:(i+1)*batch_size]
        cur_texts = [sample["text"] for sample in cur_samples]
        cur_text_ids = [sample["id"] for sample in cur_samples]
        for text_id, text in zip(cur_text_ids, cur_texts):
            processed_elements.append({"text": text,
                                       "topic1": "Не найдено",
                                       "proba1": 1.0})
    if i % 20 == 0 and i > 0:
        with open(f"processed/{args.number}_{nf}.json", 'w', encoding="utf8") as out:
            json.dump(processed_elements, out, indent=2, ensure_ascii=False)
        processed_elements = []
        nf += 1
        print("iter", nf)

if processed_examples:
    with open(f"processed/{args.number}_{nf}.json", 'w', encoding="utf8") as out:
        json.dump(processed_elements, out, indent=2, ensure_ascii=False)
