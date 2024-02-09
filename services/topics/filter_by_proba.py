import json
import random
import re

with open("processed_posts.json", 'r', encoding="utf8") as inp:
    samples = json.load(inp)
random.shuffle(samples)

bashkir = ["дэ", "нэ", "лэй", "кы", "кә", "ә", "тэ", "гы", "лэ", "мэ", "йы", "тыу", "бый"]

filtered_samples = []
for sample in samples:
    text = sample["text"]
    text = text.replace("\xa0", "")
    found = False
    if any([word in text.lower() for word in
            ["поздравляю", "поздравляем", "с праздником", "днем рождения", "соболезнования",
             "подписчик", "о здравии", "с юбилеем", "с новым годом", "подпишитесь",
             "скорейшего выздоровления", "сновым годом", "с рождеством", "новым 2024 годом", "звоните",
             "пишите", "бесплатная доставка", "скидки", "скидка", "скидкой", "смотрите", "гибкие цены",
             "по всем вопросам", "белая заработанная плата", "оформление по тк рф", "открыты вакансии",
             "официальное трудоустройство", "обязанности сотрудника", "приглашает на работу",
             "приглашаем на работу", "все вопросы по телефону", "купить билет по ссылке",
             "торопитесь закупиться", "выкупаем", "продаётся", "предлагаем вашему вниманию",
             "цены от", "цена от", "новых клиентов", "внимание клиентов", "приглашаем в команду",
             "подробнее о сотрудничестве", "предлагаю вашему вниманию", "https://vk.com/app", "жaнр:",
             "в коллекции представлены", "тренд сезона", "слушать онлайн", "стоимость от", "открывает запись",
             "открываем запись", "принимаем заказы", "принимаю заказы", "получи подарок", "альбом: https", "₽",
             "https://vk.com/album", "открываю запись", "открываем запись", "записаться на участие",
             "рады видеть", "по акции", "вы можете приобрести", "рассрочка", "на одном дыхании",
             "продажа билетов"]]):
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
    if sample["topic3"].lower() == "реклама" and sample["proba3"] > 0.95:
        found = True
    if not found and sample["proba1"] > 0.7 and sample["proba2"] > 0.1 \
            and sample["topic1"].lower() != "не найдено" and len(text) > 20:
        filtered_samples.append(sample)
print(len(filtered_samples))

with open("examples0.7.txt", 'w') as out:
    for sample in filtered_samples[:1000]:
        out.write(str(sample)+'\n\n')

with open("filtered_posts.json", 'w', encoding="utf8") as out:
    json.dump(filtered_samples, out, indent=2, ensure_ascii=False)
print("finished")