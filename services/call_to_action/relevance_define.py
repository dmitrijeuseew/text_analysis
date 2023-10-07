import json
import re
from nltk import sent_tokenize


apps = ["приложении", "приложение"]
games1 = [re.compile(r"участву(й|ем)(те)?(.*?) игр(а|е|у)"),
          re.compile(r"давай(те)?(.*?) игр(а|е|у)"),
          re.compile(r"за(ходи|йди|ходим)(те)?(.*?) игр(а|е|у)"),
          re.compile(r"попробу(ем|й)(те)?(.*?) (сыграть|играть)")
          ]
games2 = [re.compile(r"про(йти|ходите) уровень"),
          re.compile(r"получай(те)? бонус"),
          re.compile(r"(cкачай|скачивай)(те)? (приложение|игру|фото|видео|файл|бесплатно)")
         ]
groups1 = [re.compile(r"(жми|добавляйтесь|вступайте|присоединяйтесь|добавляйся|вступай|присоединяйся|присоединиться)(.*?) "
                      r"(сайт|паблик|групп|сообществ|http://)"),
           re.compile(r"пере(ходи|йди)(те)?(.*?) (ссылке)"),
           re.compile(r"(скинь|пришли|присылай)(те)?(.*?) (лс|л/с)"),
           re.compile(r"ставь(те)?(.*?) (лайк|💗)"),
           re.compile(r"(добавляй|добавь)(те)?(.*?) (себе)"),
           re.compile(r"смотри(те)?(.*?) (тут)"),
           re.compile(r"пиши(те)?(.*?) (ссылке)"),
           re.compile(r"смотри(те)?(.*?) (в вк)"),
           re.compile(r"напиши(те)?(.*?) (комментариях)")
           ]
groups2 = [re.compile(r"(нужно|надо) (войти|зарегистрироваться)"),
           re.compile(r"(подписывайтесь|подпишитесь|подписывайся|подпишись)"),
           re.compile(r"подели(те)?сь"),
           re.compile(r"(ознакомьтесь|знакомься|знакомьтесь) с"),
           re.compile(r"(тест тут|мегатест)"),
           re.compile(r"лайкни(те)?"),
           re.compile(r"заходи(те)? сюда")
          ]
adverts1 = [re.compile(r"(пишите|пиши|звони)(.*?) (вацап|ватсап|whatsapp|телеграм|тг|telegram|личку|лс|л/с|л с)"),
           re.compile(r"посмотри(те)?(.*?) (объявление)"),
           re.compile(r"размещай(те)?(.*?) (продукцию|товар)")
           ]
adverts2 = [re.compile(r"забирай(те)? тут"),
            re.compile(r"спешите к нам"),
            re.compile(r"в продаже имеется"),
            re.compile(r"(узнавайте новости|узнай)"),
           ]
other1 = [re.compile(r"береги(те)? (.*?)(себя|своих близких)"),
          re.compile(r"(нужно|надо)(.*?) (делать|сделать|посмотреть)")
          ]
other2 = [re.compile(r"будь(те)? здоровы"),
          re.compile(r"соблюдай(те)? правила"),
          re.compile(r"пристегни(те)?"),
          re.compile(r"не забывай"),
          re.compile(r"познай(те)? себя"),
          re.compile(r"нужно задуматься"),
          re.compile(r"спи(те)? спокойно"),
          re.compile(r"поздравля(ем|ю)(!| c )"),
          re.compile(r"только послушай(те)?"),
          re.compile(r"подскажи(те)?(,)? пожалуйста"),
          re.compile(r"(с днём рождения|с днем рождения|с новым годом)")
          ]


def relevance_define(text, call, found_elements):
    text_lower = text.lower().replace("   ", " ").replace("  ", " ")
    try:
        text_lines = text.split("\n")
    except:
        print(text)
    text_lines = [line.strip() for line in text_lines]
    text_lines = [line for line in text_lines if len(line) > 1]

    is_poem_or_list, is_app, is_game_group_advert_other = False, False, False

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

    for pattern_list in [games1, groups1, adverts1, other1]:
        for pattern in pattern_list:
            fnd = re.findall(pattern, text_lower)
            if fnd and len(fnd[0]) > 1 and len(fnd[0][-2].split()) < 3:
                is_game_group_advert_other = True
                break

    for pattern_list in [games2, groups2, adverts2, other2]:
        for pattern in pattern_list:
            if re.findall(pattern, text_lower):
                is_game_group_advert_other = True
                break

    if text_lower.endswith("пишите"):
        is_game_group_advert_other = True

    is_relevant = False
    if any([is_poem_or_list, is_app, is_game_group_advert_other]):
        pass
    elif call == "call" and len(text_lines) < 15:
        all_needs = True
        for found_element in found_elements:
            if found_element["criteria"] != "need":
                all_needs = False
        if all_needs:
            exclaim_elements = []
            filtered_elements = []
            for found_element in found_elements:
                sentence = found_element["sentence"]
                if sentence.endswith("?") or sentence.endswith("..."):
                    filtered_elements.append(found_element)
                elif sentence.endswith("!"):
                    exclaim_elements.append(found_element)
            if not filtered_elements:
                is_relevant = False
            elif not exclaim_elements and len(text_lines) > 5:
                is_relevant = False
            else:
                is_relevant = True
        else:
            is_relevant = True
    
    return is_relevant
