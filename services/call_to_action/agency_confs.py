import json
from deeppavlov import configs, build_model


with open("calls_2k.json", 'r') as inp:
    samples = json.load(inp)

model = build_model(configs.classifiers.agency_cls, download=False)

batch_size = 60
num_batches = len(samples) // batch_size + int(len(samples) % batch_size > 0)

texts_with_probas = []
for i in range(num_batches):
    cur_texts = samples[i*batch_size:(i+1)*batch_size]
    *_, probas = model(cur_texts)
    for text, proba in zip(cur_texts, probas):
        texts_with_probas.append([text, float(proba[1])])
    print("iter", i)

with open("texts_with_probas.json", 'w', encoding="utf8") as out:
    json.dump(texts_with_probas, out, indent=2, ensure_ascii=False)

print("finished")
