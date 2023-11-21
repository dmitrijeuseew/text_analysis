import json
import os
import pickle
import re
import cld3
import pymorphy2
import spacy
from deeppavlov import build_model
from nltk import sent_tokenize
from relevance_define import relevance_define


FIND_BY_VERBS = False

class CallToAction:
    def __init__(self, config_name):
        self.cls_model = build_model(config_name, download=True)
        self.nlp = spacy.load("ru_core_news_sm")
        self.morph = pymorphy2.MorphAnalyzer()
        self.rgx = re.compile("([\w']{2,})")

    def __call__(self, sample):
        if not isinstance(sample["text"], str):
            sample["text"] = ""
        text = sample["text"]
        found_elements = []
        if FIND_BY_VERBS:
            cls_init, found_elements = self.find_by_verbs(text)
        else:
            cls_init = "call"
        found_add_features = self.find_by_additional_features(sample)
        cls_batch, probas = self.cls_model([text])
        is_relevant = relevance_define(text, cls_init, found_elements)
        if cls_init == "call" or found_add_features:
            if is_relevant:
                if probas[0][2] > 0.5:
                    return "call", cls_init, cls_batch[0], is_relevant, found_elements
            return "not_call", cls_init, cls_batch[0], is_relevant, found_elements
        return cls_init, cls_init, cls, is_relevant, found_elements

    def find_by_additional_features(self, sample):
        found = False
        text = sample.get("text", "")
        likes = sample.get("likes", 0)
        shares = sample.get("shares", 0)
        comments = sample.get("comments", 0)
        if len(text.split()) > 30:
            found = True
        doc = self.nlp(text)
        if len(list(doc.ents)) > 0:
            found = True
        if likes or shares or comments:
            found = True
        return found

    def find_by_verbs(self, text):
        langs = ["ru", "bg"]
        try:
            lang = cld3.get_language(text)
            if lang is not None:
                lang = lang[0]
                if lang in langs:
                    found_elements = []
                    lines = text.split("\n")
                    lines = [line.strip() for line in lines if len(line.strip()) > 1]
                    sentences = []
                    for line in lines:
                        sentences += sent_tokenize(line)
                    for sentence in sentences:
                        mood_words = self.check_mood(sentence)
                        if mood_words:
                            found_elements.append({"criteria": "mood", "words": mood_words, "sentence": sentence})
                        need_words = self.check_need_word(sentence)
                        if need_words:
                            found_elements.append({"criteria": "need", "words": need_words, "sentence": sentence})
                        action_word = self.check_action_words(sentence)
                        if action_word:
                            found_elements.append({"criteria": "action_word", "words": action_word, "sentence": sentence})
                    
                    if found_elements:
                        return "call", found_elements  # action
                    else:
                        return "not_call", found_elements  # not action
        except Exception as e:
            print("error", e)
        return "undefined", []  # not russian

    def check_mood(self, text):
        found_words = []
        words = self.rgx.findall(text)
        for word in words:
            p = self.morph.parse(word)[0]
            if p.tag.mood == "impr":
                found_words.append(word)
        return found_words

    def check_need_word(self, text):
        need = ["нужно", "надо"]
        txt = text.lower()
        doc = self.nlp(txt)
        for token in doc:
            if token.head.text.lower() in need and self.morph.parse(token.text)[0].tag.POS == "INFN":
                return [token.head.text, token.text]
        return []

    def check_action_words(self, text):
        action_words = ["давайте", "предлагаю", "давно уже пора"]
        for word in action_words:
            if word in text.lower():
                return word
        return ""


text = "если хотите чтоб катались где-то в другом месте сделайте специальное место или трек для мотоспорта"
call_to_action = CallToAction("agency_cls.json")

cls = call_to_action({"text": text})
print(cls)
