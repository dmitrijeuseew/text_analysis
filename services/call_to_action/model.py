import os
import pickle
import re
import cld3
import pymorphy2
import spacy
from urllib.request import urlretrieve
from relevance_define import relevance_define


class CallToAction:
    def __init__(self, vectorizer_flname, vectorizer_url, svc_flname, svc_url):
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists(f"data/{vectorizer_flname}"):
            urlretrieve(vectorizer_url, f"data/{vectorizer_flname}")
        if not os.path.exists(f"data/{svc_flname}"):
            urlretrieve(svc_url, f"data/{svc_flname}")
        with open(f"data/{vectorizer_flname}", 'rb') as inp:
            self.tfidf_vectorizer = pickle.load(inp)
        with open(f"data/{svc_flname}", 'rb') as inp:
            self.svc = pickle.load(inp)
        self.nlp = spacy.load("ru_core_news_sm")
        self.morph = pymorphy2.MorphAnalyzer()
        self.rgx = re.compile("([\w']{2,})")

    def __call__(self, text):
        cls_init = self.find_by_verbs(text)
        if cls_init == "call":
            is_relevant = relevance_define(text, cls_init)
            if is_relevant:
                cls = self.find_class(text)
                if cls == 0:
                    return "call"
            return "not a call"
        return cls_init

    def find_by_verbs(self, text):
        langs = ["ru", "bg"]
        lang = cld3.get_language(text)
        if lang is not None:
            lang = lang[0]
            if lang in langs:
                if (
                    self.check_action_words(text)
                    or self.check_mood(text)
                    or self.check_need_word(text)
                ):
                    return "call"  # action
                else:
                    return "not a call"  # not action
        return "undefined"  # not russian

    def find_class(self, text):
        matrix = self.tfidf_vectorizer.transform([text])
        dense_matrix = matrix.toarray()
        cls = self.svc.predict(dense_matrix)
        return cls

    def check_mood(self, text):
        words = self.rgx.findall(text)
        for word in words:
            p = self.morph.parse(word)[0]
            if p.tag.mood == "impr":
                return True
        return False

    def check_need_word(self, text):
        need = ["нужно", "надо"]
        n1 = need[0].lower()
        n2 = need[1].lower()
        verb_tags = ["VERB", "MD", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "VAFIN", "VMFIN", "VVFIN"]
        txt = text.lower()
        doc = self.nlp(txt)

        for token in doc:
            if (
                (token.head.text == n1 or token.head.text == n1)
                or (token.head.text == n2 or token.head.text == n2)
            ) and token.tag_ in verb_tags:
                try:
                    if token.head.nbor(-1).text == "не":
                        continue
                except:
                    print("not found")
                return True
        return False

    def check_action_words(self, text):
        action_words = ["давайте", "предлагаю", "давно уже пора"]
        for word in action_words:
            str = "(\A|\W)" + word.lower() + "(\W|\Z)"
            s = re.search(str, text.lower())
            if s is not None:
                return True
        return False


text = "если хотите чтоб катались где-то в другом месте сделайте специальное место или трек для мотоспорта"
call_to_action = CallToAction(vectorizer_flname = "ca_vectorizer.pk",
                              vectorizer_url = "http://files.deeppavlov.ai/tmp/ca_vectorizer.pk",
                              svc_flname = "ca_svc.pk",
                              svc_url = "http://files.deeppavlov.ai/tmp/ca_svc.pk")

cls = call_to_action(text)
print(cls)
