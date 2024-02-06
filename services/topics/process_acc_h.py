import argparse
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
parser.add_argument("-t", action="store", dest="datatype")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

batch_size = 60

nf = 0
files = os.listdir(f"{args.datatype}_january1")
files = [flname for flname in files if int(flname[0]) == int(args.number)]
data = []
for flname in files:
    with open(f"{args.datatype}_january1/{flname}", 'r', encoding="utf8") as inp:
        cur_data = json.load(inp)
    for sample in cur_data:
        if sample["topic1"] == "Здравоохранение":
            data.append(sample)


topics2_h_config = parse_config("topics2_h_distil.json")
topics3_h_config = parse_config("topics3_h.json")

#cwd = os.getcwd()
cwd = "/archive/evseev/.deeppavlov/data"

for key in topics2_h_config["metadata"]["variables"].keys():
    topics2_h_config["metadata"]["variables"][key] = \
        topics2_h_config["metadata"]["variables"][key].replace("/data", cwd)
for i in range(len(topics2_h_config["metadata"]["download"])):
    topics2_h_config["metadata"]["download"][i]["subdir"] = \
        topics2_h_config["metadata"]["download"][i]["subdir"].replace("/data", cwd)
for i in range(len(topics2_h_config["chainer"]["pipe"])):
    for key in topics2_h_config["chainer"]["pipe"][i]:
        if key in ["vocab_file", "save_path", "load_path", "pretrained_bert"]:
            topics2_h_config["chainer"]["pipe"][i][key] = \
                topics2_h_config["chainer"]["pipe"][i][key].replace("/data", cwd)

for key in topics3_h_config["metadata"]["variables"].keys():
    topics3_h_config["metadata"]["variables"][key] = \
        topics3_h_config["metadata"]["variables"][key].replace("/data", cwd)
for i in range(len(topics3_h_config["metadata"]["download"])):
    topics3_h_config["metadata"]["download"][i]["subdir"] = \
        topics3_h_config["metadata"]["download"][i]["subdir"].replace("/data", cwd)
for i in range(len(topics3_h_config["chainer"]["pipe"])):
    for key in topics3_h_config["chainer"]["pipe"][i]:
        if key in ["vocab_file", "save_path", "load_path", "pretrained_bert"]:
            topics3_h_config["chainer"]["pipe"][i][key] = \
                topics3_h_config["chainer"]["pipe"][i][key].replace("/data", cwd)

topics2_h = build_model(topics2_h_config, download=True)
topics3_h = build_model(topics3_h_config, download=True)

topics2_h_list = []
with open(f"{cwd}/models/classifiers/topics2_h_distil/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics2_h_list.append(line_split[0])

topics3_h_list = []
with open(f"{cwd}/models/classifiers/topics3_h/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics3_h_list.append(line_split[0])

with open(f"{cwd}/models/classifiers/topics_by_categories.json", 'r') as inp:
    topics_by_categories = json.load(inp)

for key in topics_by_categories:
    topics_by_categories[key] = [element.strip('"').strip('\\').strip('"')
                                 for element in topics_by_categories[key]]

file_subtopics = f"{cwd}/downloads/topic_subtopics.json"
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
    num_bs = 0
    for letter in bashkir_letters:
        num_bs += text.count(letter)
        if num_bs > 2:
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
            labels2_h, probas2_h = topics2_h(raw_texts)
            labels3_h, probas3_h = topics3_h(raw_texts)

            for nl, (txt, label2_h, proba2_h, label3_h, proba3_h) in \
                    enumerate(zip(raw_texts, labels2_h, probas2_h, labels3_h, probas3_h)):
                label2_probas_h = zip(topics2_h_list, proba2_h)
                label2_probas_h = sorted(label2_probas_h, key=lambda x: x[1], reverse=True)

                label2 = label2_h
                proba2 = label2_probas_h[0][1]

                label3_probas_h = zip(topics3_h_list, proba3_h)
                label3_probas_h = sorted(label3_probas_h, key=lambda x: x[1], reverse=True)

                label3 = label3_h
                proba3 = label3_probas_h[0][1]

                if label2 == "Не найдено":
                    label3 = "Не найдено"
                    proba3 = 1.0

                f_labels.append([(label2, proba2), (label3, proba3), f_texts[nl][1]])

        for _, n in nf_texts:
            nf_labels.append([("Не найдено", 1.0), ("Не найдено", 1.0), n])

        total_labels = f_labels + nf_labels
        total_labels = sorted(total_labels, key=lambda x: x[-1])

        cur_twl = [[sample, element[0], element[1]]
                   for sample, element in zip(cur_samples, total_labels)]

        for sample, (label2, proba2), (label3, proba3) in cur_twl:
            sample["topic2"] = label2
            sample["proba2"] = float(proba2)
            sample["topic3"] = label3
            sample["proba3"] = float(proba3)
            processed_elements.append(sample)
    except Exception as e:
        print(f"error: {e}")
        cur_samples = data[i*batch_size:(i+1)*batch_size]
        for sample in cur_samples:
            sample["topic2"] = "Не найдено"
            sample["proba2"] = 1.0
            sample["topic3"] = "Не найдено"
            sample["proba3"] = 1.0
            processed_elements.append(sample)
    if i % 100 == 0 and i > 0:
        with open(f"{args.datatype}_january_h/{args.number}_{nf}.json", 'w', encoding="utf8") as out:
            json.dump(processed_elements, out, indent=2, ensure_ascii=False)
        processed_elements = []
        nf += 1
        print("iter", nf)

if processed_elements:
    with open(f"{args.datatype}_january_h/{args.number}_{nf}.json", 'w', encoding="utf8") as out:
        json.dump(processed_elements, out, indent=2, ensure_ascii=False)