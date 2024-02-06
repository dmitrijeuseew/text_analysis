import os
import json

comments = []
for d in ["comments_january_h", "comments_january_es", "comments_january_other"]:
    files = os.listdir(d)
    for flname in files:
        with open(f"{d}/{flname}", 'r') as inp:
            data = json.load(inp)
        for element in data:
            if len(element["proc_text"].split()) < 4:
                proba1 = min(0.01, element["proba1"])
                proba2 = min(0.01, element["proba2"])
                proba3 = min(0.01, element["proba3"])
            else:
                proba1 = element["proba1"]
                proba2 = element["proba2"]
                proba3 = element["proba3"]
            topic1 = element["topic1"]
            topic2 = element["topic2"]
            topic3 = element["topic3"]
            text_id = element["id"]
            text = element["text"]
            comments.append({"id": text_id, "text": text, "topic1": topic1, "proba1": proba1,
                             "topic2": topic2, "proba2": proba2, "topic3": topic3, "proba3": proba3,
                             })

print("comments", len(comments))
with open("processed_comments.json", 'w', encoding="utf8") as out:
    json.dump(comments, out, indent=2, ensure_ascii=False)

with open("processed_comments_test.json", 'w', encoding="utf8") as out:
    json.dump(comments[:2000], out, indent=2, ensure_ascii=False)


posts = []
for d in ["posts_january_h", "posts_january_es", "posts_january_other"]:
    files = os.listdir(d)
    for flname in files:
        with open(f"{d}/{flname}", 'r') as inp:
            data = json.load(inp)
        for element in data:
            if len(element["proc_text"].split()) < 4:
                proba1 = min(0.01, element["proba1"])
                proba2 = min(0.01, element["proba2"])
                proba3 = min(0.01, element["proba3"])
            else:
                proba1 = element["proba1"]
                proba2 = element["proba2"]
                proba3 = element["proba3"]
            topic1 = element["topic1"]
            topic2 = element["topic2"]
            topic3 = element["topic3"]
            text_id = element["id"]
            text = element["text"]
            posts.append({"id": text_id, "text": text, "topic1": topic1, "proba1": proba1,
                          "topic2": topic2, "proba2": proba2, "topic3": topic3, "proba3": proba3,
                          })

print("posts", len(posts))
with open("processed_posts.json", 'w', encoding="utf8") as out:
    json.dump(posts, out, indent=2, ensure_ascii=False)