import argparse
import copy
import json
import os
import re
import pickle
from logging import getLogger
from typing import Optional, List
from deeppavlov import build_model
from deeppavlov.core.commands.utils import parse_config


parser = argparse.ArgumentParser()
parser.add_argument("-n", action="store", dest="number")
parser.add_argument("-d", action="store", dest="device")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

batch_size = 60

text_type = "posts"
nf = 0
#with open(f"{text_type}_january/{args.number}.pickle", 'rb') as inp:
#    data = pickle.load(inp)
with open(f"{text_type}_january/{args.number}.json", 'r', encoding="utf8") as inp:
    data = json.load(inp)

topic_cls_config = parse_config("topic_cls_chatgpt_22.json")
topic_cls_h_e_s_config = parse_config("topic_cls_h_e_s.json")

#cwd = os.getcwd()
cwd = "/archive/evseev/.deeppavlov"
for key in topic_cls_config["metadata"]["variables"].keys():
    topic_cls_config["metadata"]["variables"][key] = \
        topic_cls_config["metadata"]["variables"][key].replace("/data", f"{cwd}/data")
for i in range(len(topic_cls_config["metadata"]["download"])):
    topic_cls_config["metadata"]["download"][i]["subdir"] = \
        topic_cls_config["metadata"]["download"][i]["subdir"].replace("/data", f"{cwd}/data")
for i in range(len(topic_cls_config["chainer"]["pipe"])):
    for key in topic_cls_config["chainer"]["pipe"][i]:
        if key in ["vocab_file", "save_path", "load_path", "pretrained_bert"]:
            topic_cls_config["chainer"]["pipe"][i][key] = \
                topic_cls_config["chainer"]["pipe"][i][key].replace("/data", f"{cwd}/data")

for key in topic_cls_h_e_s_config["metadata"]["variables"].keys():
    topic_cls_h_e_s_config["metadata"]["variables"][key] = \
        topic_cls_h_e_s_config["metadata"]["variables"][key].replace("/data", f"{cwd}/data")
for i in range(len(topic_cls_h_e_s_config["metadata"]["download"])):
    topic_cls_h_e_s_config["metadata"]["download"][i]["subdir"] = \
        topic_cls_h_e_s_config["metadata"]["download"][i]["subdir"].replace("/data", f"{cwd}/data")
for i in range(len(topic_cls_h_e_s_config["chainer"]["pipe"])):
    for key in topic_cls_h_e_s_config["chainer"]["pipe"][i]:
        if key in ["vocab_file", "save_path", "load_path", "pretrained_bert"]:
            topic_cls_h_e_s_config["chainer"]["pipe"][i][key] = \
                topic_cls_h_e_s_config["chainer"]["pipe"][i][key].replace("/data", f"{cwd}/data")

topic_cls = build_model(topic_cls_config, download=True)
topic_cls_h_e_s = build_model(topic_cls_h_e_s_config, download=True)
print("-"*20, "started")

topics1_list = []
with open(f"{cwd}/data/models/topic_cls_chatgpt_base/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics1_list.append(line_split[0])

topics1_h_e_s_list = []
with open(f"{cwd}/data/models/classifiers/topic_cls_chatgpt_base_h_e_s/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics1_h_e_s_list.append(line_split[0])

file_subtopics = f"{cwd}/data/downloads/topic_subtopics.json"
subtopics_dict = json.load(open(file_subtopics))

bashkir_letters = "ҙҡәҫңғ"
alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя1234567890.,!?;:'%()-«» " + '"' + '/'
alphabet_full = alphabet + "abcdefghijklmnopqrstuvwxyz"


def preprocess_text(text):
    found = False
    num_bs = 0
    for letter in bashkir_letters:
        num_bs += text.count(letter)
        if num_bs > 2:
            found = True
            break
    if not found:
        fnd = re.findall(r"\[(.*?)\]", text)
        if fnd:
            replace_element = "[" + fnd[0] + "]"
            text = text.replace(replace_element, " ")
        text = text.replace("\n", " ").replace("   ", " ").replace("  ", " ").strip()
        text = ''.join([symb for symb in text if symb.lower() in alphabet_full])
        text_split = text.split()
        text_split = [tok for tok in text_split if "http" not in tok and "#" not in tok and ".ru/" not in tok]
        if len(text_split) > 7:
            text_split = text_split[:150]
            text = " ".join(text_split)
            return text, found
    return text, found


processed_elements = []
num_batches = len(data) // batch_size + int(len(data) % batch_size > 0)
for i in range(num_batches):
    try:
        f_texts, nf_texts = [], []
        cur_samples = data[i*batch_size:(i+1)*batch_size]
        cur_texts = [element["text"] for element in cur_samples]
        cur_text_ids = [element["id"] for element in cur_samples]
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

            for nl, (txt, label, proba, label_h_e_s, proba_h_e_s) in \
                    enumerate(zip(raw_texts, labels, probas, labels_h_e_s, probas_h_e_s)):

                label1_probas = list(zip(topics1_list, proba))
                label1_dict = {lbl: pr for lbl, pr in label1_probas}
                label1_probas = sorted(label1_probas, key=lambda x: x[1], reverse=True)
                
                label1 = label
                proba1 = label1_probas[0][1]
                label_init = copy.deepcopy(label)
                proba_init = label1_probas[0][1]
                label1_h_e_s_probas = zip(topics1_h_e_s_list, proba_h_e_s)
                label1_h_e_s_probas = sorted(label1_h_e_s_probas, key=lambda x: x[1], reverse=True)
                if label1_h_e_s_probas[0][1] > 0.95 and label1_h_e_s_probas[0][0].lower() != "другие темы" \
                        and len(txt.strip().split()) > 2 and label1_dict[label_h_e_s] > 0.1 \
                        and label1.lower() in ["не найдено", "здравоохранение", "образование",
                                               "социальное обеспечение, защита семьи и детства"]:
                    label1 = label_h_e_s
                    proba1 = label1_dict[label1]
                elif label1.lower() == "не найдено":
                    label1 = label1_probas[1][0]
                    proba1 = label1_probas[1][1]

                f_labels.append([txt, (label1, proba1), (label_init, proba_init), f_texts[nl][1]])
        for txt, n in nf_texts:
            nf_labels.append([txt, ("Не найдено", 1.0), ("Не найдено", 1.0), n])

        total_labels = f_labels + nf_labels
        total_labels = sorted(total_labels, key=lambda x: x[-1])

        cur_twl = [[text_id, text, element[0], element[1], element[2]]
                   for text_id, text, element in zip(cur_text_ids, cur_texts, total_labels)]

        for text_id, text, proc_text, (label1, proba1), (label_init, proba_init) in cur_twl:
            processed_elements.append({"id": text_id, "text": text, "proc_text": proc_text,
                                       "topic1": label1, "topic_init": label_init,
                                       "proba1": float(proba1), "proba_init": float(proba_init)})
    except Exception as e:
        print(f"error: {e}")
        cur_samples = data[i*batch_size:(i+1)*batch_size]
        for sample in cur_samples:
            processed_elements.append({"id": sample["id"],
                                       "text": sample["text"],
                                       "topic1": "Не найдено",
                                       "proba1": 1.0})
    if i % 100 == 0 and i > 0:
        with open(f"{text_type}_january1/{args.number}_{nf}.json", 'w', encoding="utf8") as out:
            json.dump(processed_elements, out, indent=2, ensure_ascii=False)
        processed_elements = []
        nf += 1
        if int(args.number) == 0:
            print("iter", nf)

if processed_elements:
    with open(f"{text_type}_january1/{args.number}_{nf}.json", 'w', encoding="utf8") as out:
        json.dump(processed_elements, out, indent=2, ensure_ascii=False)