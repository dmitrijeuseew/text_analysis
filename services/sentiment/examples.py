import requests


url = "http://0.0.0.0:8003/get_analysis"
text = "В Уфе с 26 по 27 мая 2022 года состоится Всероссийская практическая конференция по вопросам "\
       "развития промышленных кластеров на примере Республики Башкортостан."
res = requests.post(
    url,
    json={"data": {"id": [1], "text": [text]}}
)

print(res.json())
