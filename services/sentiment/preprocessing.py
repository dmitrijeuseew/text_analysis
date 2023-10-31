import re
from tqdm import tqdm

def find_congratulation(text):
    text = str(text)

    search = re.search('(день|дн[ё|е]м|годовщин[а-я]+) (рождени[а-я]|варенья|победы|знаний|свад[а-я]+)|\
                днюхой+|юбилеем|[\d|а-я]+-?летиием|именинами+|новым годом| нг |8 марта|победой|рождеством христовым|поздравл[а-я]+',
                       text.lower())
    if search:
        return True
    else:
        return False


def clean(text):
    text = str(text)
    text =  re.sub('[http|www].*', '', text)
    text = re.sub('[#|@][А-яЁёA-z]', '', text)

    return text


def emoji2text(text, emoji, translator):
    demojized = emoji.demojize(text)

    while re.search(':[A-z]+:', text):
        f = re.search(':[A-z]+:', text)
        emoji = f.group(0)

        try:
            if re.search('(-|\)|_)', emoji):
                text = re.sub(emoji, '', text)
            else:
                translated = translator.translate(str(emoji))
                text = text[:f.span()[0]] + translated + text[f.span()[1]:]
        except:
            pass

    return text


def num2label(num):
    if num == 'LABEL_0':
        return False
    elif num == 'LABEL_1':
        return True
    else:
        return None


def check_spam(message):
    message = str(message)
    try:
        message_translated = translator.translate(message)
        is_spam = num2label(pipe(message_translated)[0]['label'])
    except:
        is_spam = 'None'

    return is_spam