import json
import re
from logging import getLogger
from typing import List
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from deeppavlov import build_model


logger = getLogger(__file__)
app = FastAPI()

batch_size = 60

class Payload(BaseModel):
    texts: List[str]


topic_cls = build_model("topic_cls_chatgpt_22.json", download=True)
topic_cls_h_e_s = build_model("topic_cls_h_e_s.json", download=True)
subtopic_cls = build_model("topic_cls_chatgpt_120.json", download=True)

topics2_h = build_model("topics2_h_distil.json", download=True)
topics3_h = build_model("topics3_h.json", download=True)

topics2_e_s = build_model("topics2_e_s.json", download=True)
topics3_e_s = build_model("topics3_e_s.json", download=True)

topics1_list = []
with open("/data/models/topic_cls_chatgpt_base/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics1_list.append(line_split[0])

topics1_h_e_s_list = []
with open("/data/models/classifiers/topic_cls_chatgpt_base_h_e_s/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics1_h_e_s_list.append(line_split[0])

topics2_list = []
with open("/data/models/topic_cls_chatgpt_sp/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics2_list.append(line_split[0])

topics2_h_list = []
with open("/data/models/classifiers/topics2_h_distil/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics2_h_list.append(line_split[0])

topics3_h_list = []
with open("/data/models/classifiers/topics3_h/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics3_h_list.append(line_split[0])

topics2_e_s_list = []
with open("/data/models/classifiers/topics2_e_s/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics2_e_s_list.append(line_split[0])

topics3_e_s_list = []
with open("/data/models/classifiers/topics3_e_s/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics3_e_s_list.append(line_split[0])

with open("/data/models/classifiers/topics_by_categories.json", 'r') as inp:
    topics_by_categories = json.load(inp)

for key in topics_by_categories:
    topics_by_categories[key] = [element.strip('"').strip('\\').strip('"')
                                 for element in topics_by_categories[key]]

file_subtopics = "/data/downloads/topic_subtopics.json"
subtopics_dict = json.load(open(file_subtopics))

bashkir_letters = "ҙҡәҫңғ"
alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя1234567890.,!?;:'%()-«» " + '"' + '/'
alphabet_full = alphabet + "abcdefghijklmnopqrstuvwxyz"


def postprocess_label(label, text):
    if (" сво " in label.lower() and re.findall(r"\bсво[\.,!\?]?\b", text.lower())) \
            or not " сво " in label.lower():
        return True
    return False


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
                labels, probas = topic_cls(raw_texts)
                labels_h_e_s, probas_h_e_s = topic_cls_h_e_s(raw_texts)
                labels_sp, probas_sp = subtopic_cls(raw_texts)
                labels2_h, probas2_h = topics2_h(raw_texts)
                labels3_h, probas3_h = topics3_h(raw_texts)
                labels2_e_s, probas2_e_s = topics2_e_s(raw_texts)
                labels3_e_s, probas3_e_s = topics3_e_s(raw_texts)

                for nl, (txt, label, proba, label_h_e_s, proba_h_e_s, label2_h, proba2_h, label3_h, proba3_h,
                        label2_e_s, proba2_e_s, label3_e_s, proba3_e_s, label_sp, proba_sp) in \
                        enumerate(zip(raw_texts, labels, probas, labels_h_e_s, probas_h_e_s, labels2_h, probas2_h, labels3_h,
                                    probas3_h, labels2_e_s, probas2_e_s, labels3_e_s, probas3_e_s, labels_sp, probas_sp)):

                    label1_probas = zip(topics1_list, proba)
                    label1_probas = sorted(label1_probas, key=lambda x: x[1], reverse=True)
                    
                    label1 = label
                    label1_h_e_s_probas = zip(topics1_h_e_s_list, proba_h_e_s)
                    label1_h_e_s_probas = sorted(label1_h_e_s_probas, key=lambda x: x[1], reverse=True)
                    if label1_h_e_s_probas[0][1] > 0.95 and label1_h_e_s_probas[0][0].lower() != "другие темы":
                        label1 = label_h_e_s
                    for cur_label, cur_proba in label1_probas:
                        if cur_label == label1:
                            proba1 = (cur_proba + proba) / 2
                            break
                    label2_probas = zip(topics2_list, proba_sp)
                    label2_probas = sorted(label2_probas, key=lambda x: x[1], reverse=True)

                    label2_probas_h = zip(topics2_h_list, proba2_h)
                    label2_probas_h = sorted(label2_probas_h, key=lambda x: x[1], reverse=True)

                    label2_probas_e_s = zip(topics2_e_s_list, proba2_e_s)
                    label2_probas_e_s = sorted(label2_probas_e_s, key=lambda x: x[1], reverse=True)

                    label2 = "Не найдено"
                    proba2 = 1.0
                    if label1.lower() == "здравоохранение":
                        label2 = label2_h
                        proba2 = label2_probas_h[0][1]
                    if label1.lower() == "образование" or "социальное" in label1.lower():
                        label2 = label2_e_s
                        proba2 = label2_probas_e_s[0][1]

                    if label1.lower() == "образование" and label2_e_s.lower() not in topics_by_categories["topics2_e"]:
                        if label2_probas_e_s[1][0].lower() in topics_by_categories["topics2_e"]:
                            label2 = label2_probas_e_s[1][0]
                            proba2 = label2_probas_e_s[1][1]
                        else:
                            label2 = "Не найдено"
                            proba2 = 1.0
                    if "социальное" in label1.lower() and (label2_e_s.lower() not in topics_by_categories["topics2_s"] or \
                                                           not postprocess_label(label2_e_s, txt)):
                        label2 = "Не найдено"
                        proba2 = 1.0
                        for cur_label2, cur_proba2 in label2_probas_e_s[1:3]:
                            if cur_label2.lower() in topics_by_categories["topics2_s"] \
                                    and cur_label2.lower() != "не найдено" and postprocess_label(cur_label2, txt):
                                label2 = cur_label2
                                proba2 = cur_proba2
                                break

                    if label1.lower() not in ["здравоохранение", "образование"] and "социальное" not in label1.lower():
                        subtopics = subtopics_dict.get(label1, [])
                        fs = False
                        for subtopic in label_sp[:10]:
                            if subtopic in subtopics:
                                label2 = subtopic
                                for cur_label2, cur_proba2 in label2_probas:
                                    if label2 == cur_label2:
                                        proba2 = cur_proba2
                                        fs = True
                                        break
                            if fs:
                                break
                    label3_probas_h = zip(topics3_h_list, proba3_h)
                    label3_probas_h = sorted(label3_probas_h, key=lambda x: x[1], reverse=True)
                    
                    label3 = "Не найдено"
                    proba3 = 1.0
                    label3_probas_e_s = zip(topics3_e_s_list, proba3_e_s)
                    label3_probas_e_s = sorted(label3_probas_e_s, key=lambda x: x[1], reverse=True)

                    if label1.lower() == "здравоохранение":
                        label3 = label3_h
                        proba3 = label3_probas_h[0][1]
                    if label1.lower() == "образование" or "социальное" in label1.lower():
                        label3 = label3_e_s
                        proba3 = label3_probas_e_s[0][1]

                    if label2 == "Не найдено":
                        label3 = "Не найдено"

                    if label1.lower() == "образование" and label3_e_s.lower() not in topics_by_categories["topics3_e"]:
                        if label3_probas_e_s[1][0].lower() in topics_by_categories["topics3_e"]:
                            label3 = label3_probas_e_s[1][0]
                            proba3 = label3_probas_e_s[1][1]
                        else:
                            label3 = "Не найдено"
                            proba3 = 1.0
                    if "социальное" in label1.lower() and (label3_e_s.lower() not in topics_by_categories["topics3_s"] or \
                                                           not postprocess_label(label3_e_s, txt)):
                        label3 = "Не найдено"
                        proba3 = 1.0
                        for cur_label3, cur_proba3 in label3_probas_e_s[1:3]:
                            if cur_label3.lower() in topics_by_categories["topics3_s"] \
                                    and cur_label3.lower() != "не найдено" and postprocess_label(cur_label3, txt):
                                label3 = cur_label3
                                proba3 = cur_proba3
                                break

                    f_labels.append([(label1, proba1), (label2, proba2), (label3, proba3), f_texts[nl][1]])

            for _, n in nf_texts:
                nf_labels.append([("Не найдено", 1.0), ("Не найдено", 1.0), ("Не найдено", 1.0), n])

            total_labels = f_labels + nf_labels
            total_labels = sorted(total_labels, key=lambda x: x[-1])

            cur_twl = [[text, element[0], element[1], element[2]] for text, element in zip(cur_texts, total_labels)]

            for text, (label1, proba1), (label2, proba2), (label3, proba3) in cur_twl:
                processed_elements.append({"text": text, "topic1": label1, "topic2": label2, "topic3": label3,
                                            "proba1": float(proba1), "proba2": float(proba2), "proba3": float(proba3)})
        except Exception as e:
            print(f"error: {e}")
            cur_texts = texts[i*batch_size:(i+1)*batch_size]
            for text in cur_texts:
                processed_elements.append({"text": text,
                                           "topic1": "Не найдено",
                                           "topic2": "Не найдено",
                                           "topic3": "Не найдено",
                                           "proba1": 1.0,
                                           "proba2": 1.0,
                                           "proba3": 1.0})
    return processed_elements


uvicorn.run(app, host='0.0.0.0', port=8002)
