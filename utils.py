import os
from constants import LANGUAGES_TO_USE


def get_english_word_filenames(english_word, languages_to_use=LANGUAGES_TO_USE, print_alerts=True):
    out_filenames = {}
    english_filenames = os.listdir("./audio/english")
    target_english_filename = ""
    times_found = 0  # ensure a word doesn't appear twice
    for filename in english_filenames:
        cur_word = filename.split("-")[-1][:-4]
        # print(cur_word)
        if english_word == cur_word:
            target_english_filename = "./audio/english/" + filename
            times_found += 1

    if times_found == 0:
        if print_alerts:
            print("WORD NOT FOUND")
        return 404
    # assert (times_found == 1)

    # print("target english filename:", target_english_filename)

    target_index = int((target_english_filename.split("/")[-1]).split("-")[0])
    # print(target_index)

    for language in languages_to_use:
        cur_filenames = os.listdir("./audio/" + language)
        if len(cur_filenames) == 0:
            if print_alerts:
                print("No files for", language, "skipping:", english_word)
            continue
        for filename in cur_filenames:
            if filename.split(".")[-1] != "mp3":
                continue
            filename_index = int(filename.split("-")[0])
            if filename_index == target_index:
                out_filenames[language] = "./audio/" + language + "/" + filename
                break

    return out_filenames


if __name__ == "__main__":
    print(get_english_word_filenames("work", print_alerts=False))
    print(get_english_word_filenames("chair"))
