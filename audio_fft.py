# import pydub
import numpy as np
# from scipy.io import wavfile
from audio2numpy import open_audio
import scipy
from constants import LANGUAGES_TO_USE
from scipy import signal
import matplotlib.pyplot as plt
import os

from meaning_similarity import get_meaning_diff
from utils import get_english_word_filenames
from translate import load_common_words
import csv
import statistics


def read(filename, normalized=False):
    signal, sampling_rate = open_audio(filename)

    return signal, sampling_rate
    # fs, data = wavfile.read(filename)
    # return fs, data
    # """MP3 to numpy array"""
    # a = pydub.AudioSegment.from_mp3(f)
    # y = np.array(a.get_array_of_samples())
    # if a.channels == 2:
    #     y = y.reshape((-1, 2))
    # if normalized:
    #     return a.frame_rate, np.float32(y) / 2 ** 15
    # else:
    #     return a.frame_rate, y


def generate_graph(filename, subplot_count=1):
    signals, sampling_rate = read(filename)
    print("signals shape", signals.shape)
    print("sampling rate", sampling_rate)
    transformed = np.fft.fft(signals)
    print("transformed shaped", transformed.shape)
    t = np.arange(signals.shape[0])
    freq = np.fft.fftfreq(t.shape[-1])
    plt.subplot(4, 1, subplot_count)
    plt.tight_layout()
    plt.plot(freq, transformed.real, freq, transformed.imag)
    # plt.plot(freq, transformed.real)
    plt.title((filename.split("-")[-1])[:-4])


def generate_periodogram(filename):
    signals, sampling_rate = read(filename)
    # print("sample rate", sampling_rate)
    # print("signals shape", signals.shape)
    f, Pxx_den = signal.periodogram(signals)
    # f, Pxx_den = f[:8000], Pxx_den[:8000]
    # print("f shape", f.shape)
    # print("Pxx_den shape", Pxx_den.shape)
    return f, Pxx_den


def graph_periodogram(filename, subplot_count=1):
    f, Pxx_den = generate_periodogram(filename)
    plt.subplot(4, 1, subplot_count)
    plt.tight_layout()
    plt.semilogy(f, Pxx_den)
    plt.ylim([1e-7, 1e2])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.title((filename.split("-")[-1])[:-4])
    # plt.show()


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm


def get_l2(a, b):
    # if a != b:
    a = np.asarray(a)
    b = np.asarray(b)
    # a = normalize(a)
    # b = normalize(b)
    # print(np.sum(np.abs(a - b)))

    dist = np.linalg.norm(a - b)
    return dist


def cross_correlate(a, b):
    # print("a shape", a.shape)
    # print("b shape", b.shape)
    corr = signal.correlate(a, b)
    # print("corr shape", corr.shape)
    corr = np.average(corr)
    return corr


def compare_two_words(filename_a, filename_b):
    word_a = (filename_a.split("-")[-1])[:-4]
    word_b = (filename_b.split("-")[-1])[: -4]
    f_a, Pxx_den_a = generate_periodogram(filename_a)
    f_b, Pxx_den_b = generate_periodogram(filename_b)

    a = regularize_signal_domain(f_a, Pxx_den_a)
    b = regularize_signal_domain(f_b, Pxx_den_b)

    # a = np.asarray([f_a, Pxx_den_a])
    # b = np.asarray([f_b, Pxx_den_b])

    diff = get_l2(a, b)
    print("Diff between " + word_a + " and " + word_b + ":", diff)
    return diff
    # print("Cross between f " + word_a + " and " + word_b + ":", cross_correlate(a, b))
    # print("Diff between f " + word_a + " and " + word_b + ":", get_l2(f_a, f_b))
    # print("Diff between Pxx " + word_a + " and " + word_b + ":", get_l2(Pxx_den_a, Pxx_den_b))


def regularize_signal_domain(f, Pxx_den, max_freq=.4, interval=.01):
    num_intervals = max_freq / interval
    assert ((int(num_intervals) - num_intervals) == 0)
    num_intervals = int(num_intervals)
    out = []

    cur_f_index = 0
    for cur_frequency in np.arange(0, max_freq, interval):
        cur_lowest = cur_frequency
        cur_highest = cur_frequency + interval
        cur_out = []
        for i in range(cur_f_index, f.shape[0]):
            cur_f_index = i
            if cur_lowest <= f[i] <= cur_highest:
                # calls if current frequency is in the range
                cur_out.append(Pxx_den[i])
                # cur_f_index = i + 1 # + 1 so that
                continue  # lest loop break
            break
        out.append(cur_out)

    averaged_out = []
    for window in out:
        if not window:
            print("EMPTY WINDOW")
            averaged_out.append(0)
        else:
            averaged_out.append(sum(window) / len(window))
    # print(averaged_out)
    return averaged_out


def get_sound_diff(filename_a, filename_b):
    f_a, Pxx_den_a = generate_periodogram(filename_a)
    a = regularize_signal_domain(f_a, Pxx_den_a)

    f_b, Pxx_den_b = generate_periodogram(filename_b)
    b = regularize_signal_domain(f_b, Pxx_den_b)

    diff = get_l2(a, b)
    return diff


def compare_word_with_others(word_filename, other_filenames, sort=True, print_results=False):
    differences = {}

    f_a, Pxx_den_a = generate_periodogram(word_filename)
    a = regularize_signal_domain(f_a, Pxx_den_a)
    for filename in other_filenames:
        word = (filename.split("-")[-1])[:-4]
        # print("word", word)
        f_b, Pxx_den_b = generate_periodogram(filename)
        b = regularize_signal_domain(f_b, Pxx_den_b)
        diff = get_l2(a, b)
        differences[word] = diff
        # differences[word] = (compare_two_words(word_filename, filename))

    # print("differences un sorted", differences)

    if sort:
        differences = {k: v for k, v in sorted(differences.items(), key=lambda item: item[1])}
    differences = [(k, v) for k, v in differences.items()]
    # print(difference«s)
    if print_results:
        if len(differences) > 21:
            print(differences[:10])
            print(differences[-10:])
        else:
            print(differences)
    return differences


def get_filenames_from_dir(dir_path):
    files = os.listdir(dir_path)
    out_files = []
    for file in files:
        if file.split(".")[-1] == "mp3":
            out_files.append(dir_path + file)
    return out_files


def find_diff_among_languages(english_word, languages_to_use=None):
    if languages_to_use is None:
        filenames_dict = get_english_word_filenames(english_word, print_alerts=False)
    else:
        filenames_dict = get_english_word_filenames(english_word, print_alerts=False, languages_to_use=languages_to_use)
    if filenames_dict == 404:
        print("WORD NOT FOUND")
        return None, None

    filenames = list(filenames_dict.values())
    # print(filenames[0])
    differences = []
    for i in range(0, len(filenames) - 1):
        # Loop computes differences between all combos
        cur_differences_tuple = compare_word_with_others(filenames[i], filenames[i + 1:], print_results=False,
                                                         sort=False)
        cur_differences = [x[1] for x in cur_differences_tuple]
        differences += cur_differences

    # print(differences, len(differences))
    avg_difference = sum(differences) / len(differences)
    standard_dev = statistics.stdev(differences)
    print("average difference:", avg_difference, "standard dev:", standard_dev, "---", english_word)
    # print("Difference", english_word, ":", avg_difference)
    # print()
    return avg_difference, standard_dev


def find_intralanguage_connections(language, save_filename_body):
    words = load_common_words("./translations/" + language + "_common-nouns.csv")
    # Gets the corresponding file for each word
    filenames = []
    for word in words:
        filename_dict = get_english_word_filenames(word, languages_to_use=["english"], print_alerts=False)
        if filename_dict == 404:
            print("UNABLE TO FIND", word)
            continue
        filenames.append(list(filename_dict.values())[0])

    # filenames = [list(get_english_word_filenames(word, languages_to_use=["english"], print_alerts=False).values())[0]
    #              for word in words]
    # filenames = list(filenames_dict.values())
    # print(filenames)
    stats = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            cur_word, cur_word_filename = words[i], filenames[i]
            cur_compare, cur_compare_filename = words[j], filenames[j]
            cur_meaning_diff = get_meaning_diff(cur_word, cur_compare)
            if cur_meaning_diff == 404:
                continue
            cur_sound_diff = get_sound_diff(cur_word_filename, cur_compare_filename)
            cur_stat = (cur_word, cur_compare, cur_sound_diff, cur_meaning_diff)
            # print("cur stat", cur_stat)
            stats.append(cur_stat)

    print("unsorted", stats)

    stats_sort_by_sound = sorted(stats, key=sort_stats_index_2)
    stats_sort_by_sound = [["word_a", "word_b", "meaning_diff", "sound_diff"]] + stats_sort_by_sound
    with open(save_filename_body + "_sound.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(stats_sort_by_sound)

    stats_sort_by_meaning = sorted(stats, key=sort_stats_index_3)
    stats_sort_by_meaning = [["word", "word_b", "meaning_diff", "sound_diff"]] + stats_sort_by_meaning
    with open(save_filename_body + "_meaning.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(stats_sort_by_meaning)

    stats = np.asarray(stats)[:, 2:].astype(np.float)  # 2 remove words so can take avg
    averages = np.mean(stats, axis=0)
    print("average meaning diff", averages[0])
    print("average sound diff", averages[1])


def sort_stats_index_2(x):
    return x[2]


def sort_stats_index_3(x):
    return x[3]


def sort_stats(x):
    return x[1]  # cur diff


def compare_languages(languages_to_use, save_filename):
    words = load_common_words("./translations/english_common-nouns.csv")
    differences = {}
    stats = []
    num_words = len(words)
    for i, english_word in enumerate(words):
        cur_diff, cur_sd = find_diff_among_languages(english_word, languages_to_use=languages_to_use)
        if i % 10 == 0:
            print("Progress", str(int((i / num_words) * 100)) + "%")
        if cur_diff is not None:
            differences[english_word] = cur_diff
            stats.append([english_word, cur_diff, cur_sd])

    # stats = sorted([(k, v) for k, v in stats.items()], key=sort_stats)
    stats = sorted(stats, key=sort_stats)
    stats = [["word", "diff", "sd"]] + stats
    print("stats", stats)
    differences = {k: v for k, v in sorted(differences.items(), key=lambda item: item[1])}
    differences = [(k, v) for k, v in differences.items()]

    print(differences[:10])
    print(differences[-10:])

    with open(save_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(stats)


def main():
    languages_to_use = [
        "spanish",
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
        "japanese",
        "norwegian",
    ]

    languages_to_use = LANGUAGES_TO_USE
    # compare_languages(languages_to_use, save_filename="./results/test5sa.csv")

    find_intralanguage_connections("english", "./results/test-intra-1a.csv")

    # find_diff_among_languages("person", languages_to_use=languages_to_use)

    # filenames = ["./audio/spanish/515-papá.mp3", "./audio/spanish/516-proporción.mp3",
    #              "./audio/spanish/319-parlamento.mp3", "./audio/spanish/740-fase.mp3"]
    filenames = ["./audio/english/446-relation.mp3", "./audio/english/539-organization.mp3",
                 "./audio/english/6-man.mp3", "./audio/english/26-business.mp3", "./audio/english/603-commitment.mp3",
                 "./audio/english/790-interpretation.mp3", "./audio/english/4-government.mp3"]

    # filenames = get_filenames_from_dir("./audio/english/") + get_filenames_from_dir(
    #     "./audio/spanish/") + get_filenames_from_dir("./audio/french/")

    # print(get_filenames_from_dir("./audio/english/"))

    # print("target filename", filenames[0])
    # compare_word_with_others(filenames[0], filenames[:10], print_results=True)

    # print("target filename", filenames[-1])
    # compare_word_with_others(filenames[-1], filenames)
    # f, Pxx_den = generate_periodogram(filenames[0])
    # regularize_signal_domain(f, Pxx_den)
    # for file in filenames:
    #     compare_two_words(filenames[0], file)
    # subplot_count = 1
    # for filename in filenames:
    #     # generate_graph(filename, subplot_count=subplot_count)
    #     graph_periodogram(filename, subplot_count=subplot_count)
    #     subplot_count += 1
    #
    # plt.show()


if __name__ == "__main__":
    main()
