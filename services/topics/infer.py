# !pip install deeppavlov
from deeppavlov import build_model


#model = build_model("topic_cls_chatgpt_22.json", download=True, install=True)

# Другие модели:
# model = build_model("topic_cls_chatgpt_120.json", download=True, install=True)
# model = build_model("topic_cls_llama_120.json", download=True, install=True)

bashkir_letters = "ҙҡәҫңғ"
alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя1234567890.,!?;:'%()-«» " + '"' + '/'
alphabet_full = alphabet + "abcdefghijklmnopqrstuvwxyz"

text = "В Уфе с 26 по 27 мая 2022 года состоится Всероссийская практическая конференция по вопросам развития промышленных кластеров на примере Республики Башкортостан. Главные цели конференции - раскрытие инвестиционного потенциала региона, демонстрация лучших практик, получение практических навыков по выявлению инвестиционных ниш кластеров и работе по реализации импортозамещающих проектов. "


def get_topic(text):
    found = False
    for letter in bashkir_letters:
        if letter in text:
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
            labels = model([text])
            return labels[0]
    return "Не найдено"


topic = get_topic(text)
print("topic:", topic)
