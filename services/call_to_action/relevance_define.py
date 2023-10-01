import json
import re


apps = ["приложении", "приложение"]
games = [re.compile(r"участвуй(те)? в игре"),
         re.compile(r"пройти уровень"),
         re.compile(r"заходи(те)? в игру"),
         re.compile(r"получай(те)? бонус"),
         re.compile(r"(cкачай|скачивай)(те)? (приложение|игру|фото|видео|файл|бесплатно)")
         ]
groups = [re.compile(r"(добавляйтесь|вступайте|присоединяйтесь|добавляйся|вступай|присоединяйся) в (группу|сообщество)"),
          re.compile(r"переходи(те)? по ссылке"),
          re.compile(r"(скинь|скиньте|пришли|пришлите|присылай|присылайте|) в (лс|л/с)"),
          re.compile(r"предлагаем присоединиться к (.*?) (сообществам|группам)"),
          re.compile(r"подписывайтесь"),
          re.compile(r"(ознакомьтесь|ознакомься|знакомьтесь) с"),
          ]
group_words = ["сайт", "группа", "группе", "паблик", "сообщество"]

adverts = [re.compile(r"в продаже имеется"),
           re.compile(r"(пишите|пиши|звони) (в|на) (вацап|ватсап|whatsapp|телеграм|тг|telegram|личку|лс|л/с)"),
           re.compile(r"спешите к нам"),
           re.compile(r"посмотри(те)? объявление"),
           re.compile(r"размещай(те)? (.*?) продукцию"),
           re.compile(r"(узнавайте новости|узнай)"),
           ]
other = [re.compile(r"будь(те)? здоровы"),
         re.compile(r"соблюдай(те)? правила"),
         re.compile(r"пристегни(те)?"),
         re.compile(r"береги(те)? себя"),
         re.compile(r"береги(те)? (.*?)своих близких"),
         re.compile(r"не забывай"),
         re.compile(r"познай(те)? себя")
         ]


def relevance_define(text, call):
    text_lower = text.lower()
    try:
        text_lines = text.split("\n")
    except:
        print(text)
    text_lines = [line.strip() for line in text_lines]
    text_lines = [line for line in text_lines if len(line) > 1]

    is_poem_or_list, is_app, is_game, is_group, is_advert, is_other = False, False, False, False, False, False

    num_short, equal_symb = 0, 0
    for i in range(len(text_lines)):
        if len(text_lines[i].split()) < 8:
            num_short += 1
            if num_short >= 4:
                is_poem_or_list = True
        else:
            num_short = 0

        if i > 0 and text_lines[i][0] == text_lines[i - 1][0]:
            equal_symb += 1
            if equal_symb >= 3:
                is_poem_or_list = True
        else:
            equal_symb = 0
        if is_poem_or_list:
            break

    if len(text.split()) < 10 and any([word in text_lower for word in apps]):
        is_app = True
    for pattern in games:
        if re.findall(pattern, text_lower):
            is_game = True
            break

    for pattern in groups:
        if re.findall(pattern, text_lower):
            is_group = True
            break
    if any([word in text_lower for word in ["присоединяйтесь", "присоединяйся"]]) \
            and any([word in text_lower for word in group_words]):
        is_group = True

    for pattern in adverts:
        if re.findall(pattern, text_lower):
            is_advert = True
            break
    for pattern in other:
        if re.findall(pattern, text_lower):
            is_other = True
            break

    is_relevant = False
    if any([is_poem_or_list, is_app, is_game, is_group, is_advert, is_other]):
        pass
    elif call == "call" and len(text_lines) < 15:
        is_relevant = True
    return is_relevant
