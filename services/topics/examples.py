import requests


text1 = "В Уфе с 26 по 27 мая 2022 года состоится Всероссийская практическая конференция по вопросам развития промышленных кластеров на примере Республики Башкортостан. Главные цели конференции - раскрытие инвестиционного потенциала региона, демонстрация лучших практик, получение практических навыков по выявлению инвестиционных ниш кластеров и работе по реализации импортозамещающих проектов."

text2 = "✅ Здоровая спина ✅ Реабилитация после травм любой сложности Записаться на тренировки к Екатерине вы можете на рецепции клуба, по телефону+ 7 (473) 233-13-33 или написав в директ@ ek. ssm"

res = requests.post("http://0.0.0.0:8002/model", json={"texts": [text1]})
if res.status_code == 200:
    print(res.json())
else:
    print("not found", res.status_code)