# !pip install deeppavlov
from deeppavlov import train_model


model = train_model("topic_cls_chatgpt_22.json", download=True, install=True)

# Другие модели:
# model = train_model("topic_cls_chatgpt_120.json", download=True, install=True)
