import requests


url = "http://0.0.0.0:8001"

res = requests.post(
    f"{url}/model",
    json={"entity_substr": [["москва", "россии"]],
          "entity_offsets": [[[0, 6], [17, 23]]],
          "tags": [["LOC", "LOC"]],
          "sentences_offsets": [[[0, 24]]],
          "sentences": [["Москва - столица России."]],
          "probas": [[0.42]]
          }
)

if res.status_code == 200:
    print(res.json())
