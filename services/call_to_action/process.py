import argparse
import json
import os
import pickle
from deeppavlov import build_model


parser = argparse.ArgumentParser()
parser.add_argument("-n", action="store", dest="number")
parser.add_argument("-d", action="store", dest="device")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

files = os.listdir("/home/evseev/topics/text_analysis/services/topics/texts_08.12_processed")
samples = []
for flname in files:
    if flname.startswith(f"{args.number}_"):
        with open(f"/home/evseev/topics/text_analysis/services/topics/texts_08.12_processed/{flname}", 'r') as inp:
            print(flname)
            cur_samples = json.load(inp)
        samples += cur_samples

model = build_model("agency_cls.json", download=True)

batch_size = 60
num_batches = len(samples) // batch_size + int(len(samples) % batch_size > 0)

texts_with_probas = []
for i in range(num_batches):
    cur_samples = samples[i*batch_size:(i+1)*batch_size]
    cur_texts = [sample["text"] for sample in cur_samples]
    cur_text_ids = [sample["id"] for sample in cur_samples]
    y_pred, probas = model(cur_texts)
    for sample, cls, proba in zip(cur_samples, y_pred, probas):
        sample["agency_class"] = cls
        sample["agency_proba"] = float(round(proba[2], 2))
        texts_with_probas.append(sample)
    if i % 100 == 0 and i > 0:
        print("iter", i)

with open(f"/home/evseev/topics/text_analysis/services/topics/texts_08.12_ca/{args.number}.json", 'w', encoding="utf8") as out:
    json.dump(texts_with_probas, out, indent=2, ensure_ascii=False)

print("finished")
