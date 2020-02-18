import six
from google.cloud import translate_v2 as translate
import os
import pandas as pd
from constants import LANGUAGES, LANGUAGES_TO_USE
import csv
import time

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/daniellongo/keys/lang-d5630e9c6c37.json"

translate_client = translate.Client()


def load_common_words(csv_file):
    df = pd.read_csv(csv_file)
    words = list(df["Word"])
    print("number of words", len(words))
    return words


def save_words(words, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Word"])
        for word in words:
            writer.writerow([word])


def translate_words(words, target_lang):
    translated_words = []
    for word in words:
        translated_words.append(translate_word(word, target_lang))
    return translated_words


def main():
    words = load_common_words("./translations/english_common-nouns.csv")
    # already_processed = ["spanish", "latin", "korean", "french", "english", "italian", "portuguese"]
    already_processed = ["spanish",
    "latin",
    "korean",
    "french",
    "english",
    "italian",
    "portuguese",
    "arabic",
    "czech",
    "dutch",
    "german",
    "greek",
    "hindi",
    "hungarian",
    "indonesian",
    "japanese",]
    # not_processed = ["italian", "portuguese", ]
    for language_to_use in LANGUAGES_TO_USE:
    # for language_to_use in not_processed:
        print("NOW TRANSLATING TO:", language_to_use)
        if language_to_use in already_processed:
            print("ALREADY TRANSLATED SKIPPING")
            continue
        translated_words = translate_words(words, language_to_use)
        save_words(translated_words, "./translations/" + language_to_use + "_common-nouns.csv")

    # Test
    # translate_word("hello", "latin")
    # save_words(["a", "b", "c"], "./translations/test.csv")


def translate_word(word, target_lang, input_lang='english'):
    time.sleep(.3)
    target_lang_code = LANGUAGES[target_lang]

    if input_lang is None:
        # google translate will detect input language
        input_lang_code = None
    else:
        input_lang_code = LANGUAGES[input_lang]

    text = word
    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(
        text, target_language=target_lang_code, source_language=input_lang_code)

    print(u'Text: {}'.format(result['input']))
    print(u'Translation: {}'.format(result['translatedText']))
    # print(u'Detected source language: {}'.format(
    #     result['detectedSourceLanguage']))
    return result["translatedText"]


if __name__ == "__main__":
    main()
