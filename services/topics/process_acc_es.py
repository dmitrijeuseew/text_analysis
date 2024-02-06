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
        label1 = sample["topic1"]
        if label1.lower() == "образование" or "социальное" in label1.lower():
            data.append(sample)

topics2_e_s_config = parse_config("topics2_e_s.json")
topics3_e_s_config = parse_config("topics3_e_s.json")

cwd = os.getcwd()
cwd = "/archive/evseev/.deeppavlov/data"

for key in topics2_e_s_config["metadata"]["variables"].keys():
    topics2_e_s_config["metadata"]["variables"][key] = \
        topics2_e_s_config["metadata"]["variables"][key].replace("/data", cwd)
for i in range(len(topics2_e_s_config["metadata"]["download"])):
    topics2_e_s_config["metadata"]["download"][i]["subdir"] = \
        topics2_e_s_config["metadata"]["download"][i]["subdir"].replace("/data", cwd)
for i in range(len(topics2_e_s_config["chainer"]["pipe"])):
    for key in topics2_e_s_config["chainer"]["pipe"][i]:
        if key in ["vocab_file", "save_path", "load_path", "pretrained_bert"]:
            topics2_e_s_config["chainer"]["pipe"][i][key] = \
                topics2_e_s_config["chainer"]["pipe"][i][key].replace("/data", cwd)

for key in topics3_e_s_config["metadata"]["variables"].keys():
    topics3_e_s_config["metadata"]["variables"][key] = \
        topics3_e_s_config["metadata"]["variables"][key].replace("/data", cwd)
for i in range(len(topics3_e_s_config["metadata"]["download"])):
    topics3_e_s_config["metadata"]["download"][i]["subdir"] = \
        topics3_e_s_config["metadata"]["download"][i]["subdir"].replace("/data", cwd)
for i in range(len(topics3_e_s_config["chainer"]["pipe"])):
    for key in topics3_e_s_config["chainer"]["pipe"][i]:
        if key in ["vocab_file", "save_path", "load_path", "pretrained_bert"]:
            topics3_e_s_config["chainer"]["pipe"][i][key] = \
                topics3_e_s_config["chainer"]["pipe"][i][key].replace("/data", cwd)

topics2_e_s = build_model(topics2_e_s_config, download=True)
topics3_e_s = build_model(topics3_e_s_config, download=True)


topics2_e_s_list = []
with open(f"{cwd}/models/classifiers/topics2_e_s/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics2_e_s_list.append(line_split[0])

topics3_e_s_list = []
with open(f"{cwd}/models/classifiers/topics3_e_s/classes.dict", 'r') as inp:
    lines = inp.readlines()
    for line in lines:
        line_split = line.strip().split("\t")
        topics3_e_s_list.append(line_split[0])

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
            labels2_e_s, probas2_e_s = topics2_e_s(raw_texts)
            labels3_e_s, probas3_e_s = topics3_e_s(raw_texts)

            for nl, (txt, label2_e_s, proba2_e_s, label3_e_s, proba3_e_s) in \
                    enumerate(zip(raw_texts, labels2_e_s, probas2_e_s, labels3_e_s, probas3_e_s)):

                label1 = cur_samples[f_texts[nl][1]]["topic1"]
                label2_probas_e_s = zip(topics2_e_s_list, proba2_e_s)
                label2_probas_e_s = sorted(label2_probas_e_s, key=lambda x: x[1], reverse=True)

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
                    for cur_label2, cur_proba2 in label2_probas_e_s[1:4]:
                        if cur_label2.lower() in topics_by_categories["topics2_s"] \
                                and cur_label2.lower() != "не найдено" and postprocess_label(cur_label2, txt):
                            label2 = cur_label2
                            proba2 = cur_proba2
                            break

                label3_probas_e_s = zip(topics3_e_s_list, proba3_e_s)
                label3_probas_e_s = sorted(label3_probas_e_s, key=lambda x: x[1], reverse=True)

                label3 = label3_e_s
                proba3 = label3_probas_e_s[0][1]

                if label2 == "Не найдено":
                    label3 = "Не найдено"

                if label1.lower() == "образование" and label3_e_s.lower() not in topics_by_categories["topics3_e"]:
                    label3 = "Не найдено"
                    proba3 = 1.0
                    for cur_label3, cur_proba3 in label3_probas_e_s[1:4]:
                        if cur_label3.lower() in topics_by_categories["topics3_e"] \
                                and cur_label3.lower() != "не найдено":
                            label3 = cur_label3
                            proba3 = cur_proba3
                            break
                if "социальное" in label1.lower() and (label3_e_s.lower() not in topics_by_categories["topics3_s"] or \
                                                        not postprocess_label(label3_e_s, txt)):
                    label3 = "Не найдено"
                    proba3 = 1.0
                    for cur_label3, cur_proba3 in label3_probas_e_s[1:4]:
                        if cur_label3.lower() in topics_by_categories["topics3_s"] \
                                and cur_label3.lower() != "не найдено" and postprocess_label(cur_label3, txt):
                            label3 = cur_label3
                            proba3 = cur_proba3
                            break

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
        with open(f"{args.datatype}_january_es/{args.number}_{nf}.json", 'w', encoding="utf8") as out:
            json.dump(processed_elements, out, indent=2, ensure_ascii=False)
        processed_elements = []
        nf += 1
        print("iter", nf)

if processed_elements:
    with open(f"{args.datatype}_january_es/{args.number}_{nf}.json", 'w', encoding="utf8") as out:
        json.dump(processed_elements, out, indent=2, ensure_ascii=False)