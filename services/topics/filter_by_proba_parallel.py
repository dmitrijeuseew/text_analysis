import json
import multiprocessing as mp
import os
import re

bashkir = ["дэ", "нэ", "лэй", "кы", "кә", "ә", "тэ", "гы", "лэ", "мэ", "йы", "тыу", "бый"]

with open("adv_words.txt", 'r') as inp:
    lines = inp.readlines()
adv_words = [line.strip() for line in lines]

for ii in range(4):
    files = os.listdir("all_january1")
    files = [flname for flname in files if int(flname.split("_")[0]) == ii]
    files = sorted(files, key=lambda x: int(x.split(".")[0].split("_")[-1]))

    def run(num_proc, cur_files, common_dict):
        total_samples_init = 0
        if num_proc == 0:
            print("files", len(cur_files))
        filtered_samples = []
        for flname in cur_files:
            if num_proc == 0:
                print(flname)
            with open(f"all_january1/{flname}", 'r', encoding="utf8") as inp:
                samples = json.load(inp)
                total_samples_init += len(samples)
                for sample in samples:
                    text = sample["text"]
                    text = text.replace("\xa0", "")
                    found = False
                    if any([word in text.lower() for word in adv_words]):
                        found = True
                    fnd1 = re.findall(r"[\d]{11}", text)
                    if fnd1:
                        found = True
                    fnd2 = re.findall(r"цена(.*?)руб", text.lower())
                    if fnd2 and len(fnd2[0].strip().split()) < 2:
                        found = True
                    fnd3 = re.findall(r"стоимость(.*?)руб", text.lower())
                    if fnd3 and len(fnd3[0].strip().split()) < 2:
                        found = True
                    if re.findall(r"[78][ ]?[\(]?[\d]{3}[\)]?[ ]?[-]?[\d]{3}-[\d]{2}-[\d]{2}", text):
                        found = True
                    if re.findall(r"[\d]{3,6} р\.", text):
                        found = True
                    if re.findall(r"[\d]{3,6} руб", text):
                        found = True
                    if any([letters in text for letters in bashkir]):
                        found = True
                    if any([text.lower().startswith(word) for word in
                            ["сдается", "занимаемся", "продаю", "продам", "акция", "продается", "хочешь", "нужен",
                            "ищу", "цена", "афиша"]]):
                        found = True
                    if len(text.split("\n")) > 9:
                        found = True
                    if text.count("http") > 1:
                        found = True
                    if not found and sample["proba1"] > 0.7 \
                            and sample["topic1"].lower() != "не найдено" and len(text) > 20:
                        filtered_samples.append(sample)
        common_dict[num_proc] = filtered_samples

    num_procs = 20
    chunk_size = len(files) // num_procs + int(len(files) % num_procs > 0)

    procs = []
    manager = mp.Manager()
    common_dict = manager.dict()
    for i in range(num_procs):
        proc = mp.Process(target=run, args=(i, files[i*chunk_size:(i+1)*chunk_size], common_dict))
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()
    print("finished processing")

    total_filtered_samples = []
    for i in range(num_procs):
        total_filtered_samples += common_dict[i]

    print(len(total_filtered_samples))
    with open(f"all_january1f/{ii}.json", 'w', encoding="utf8") as out:
        json.dump(total_filtered_samples, out, indent=2, ensure_ascii=False)
    print("finished", ii)
print("finished")