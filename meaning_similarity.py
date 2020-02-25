import gensim
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin', binary=True)


def get_meaning_diff(word_a, word_b):
    try:
        vector_a = model.wv[word_a]
        vector_b = model.wv[word_b]
        diff = np.sum(np.abs(vector_a - vector_b))
    except KeyError:
        print("VECTOR KEY ERROR WITH", word_a, word_b)
        return 404

    return diff


if __name__ == "__main__":
    print(
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    print(model.wv["hello"])
