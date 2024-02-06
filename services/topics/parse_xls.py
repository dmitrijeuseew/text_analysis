import json
import pandas as pd

samples = []
xls = pd.ExcelFile("post_january.xlsx")
print("read")
for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet)
    print("sheet")
    for ind, row in df.iterrows():
        samples.append({"id": row["id"], "text": row["text"]})
        if len(samples) % 50000 == 0:
            print(len(samples))

chunk_size = len(samples) // 4 + int(len(samples) % 4 > 0)
for i in range(4):
    with open(f"posts_january/{i}.json", 'w', encoding="utf8") as out:
        json.dump(samples[i*chunk_size:(i+1)*chunk_size], out, indent=2, ensure_ascii=False)

print(len(samples))