import re
import string


def prepare_text(path, is_path=True):
    if is_path:
        with open(path) as file:
            texts = file.read().split("\n\n\n")
    else:
        texts = [path]

    regex = re.compile('[^a-zA-Z]')
    texts = [regex.sub('', txt).lower() for txt in texts]

    alphabet = dict.fromkeys(string.ascii_lowercase, 0)
    texts_dict = []

    for txt in texts:
        alphabet_tmp = alphabet.copy()
        letters_count = len(list(txt))
        for char in list(txt):
            alphabet_tmp[char] += 1
        for key, value in alphabet_tmp.items():
            alphabet_tmp[key] = value / letters_count

        texts_dict.append(alphabet_tmp)

    return [list(x.values()) for x in texts_dict]
