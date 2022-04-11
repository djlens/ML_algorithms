import os
from prepare_text import prepare_text
import numpy as np
from SingleLayerPerceptron import SingleLayerPerceptron


def train():
    labels = []
    data = []
    os.chdir('./data')
    for file_name in os.listdir():
        if not file_name.startswith('.'):
            labels.append(file_name)
            os.chdir(file_name)
            data.append(prepare_text(file_name + '_texts.txt'))
            os.chdir('../')

    # generate "y" labels
    X = np.array(data)
    X = X.reshape(-1, X.shape[-1])
    y = []
    for i in range(len(data)):
        for _ in data[i]:
            y.append(i)

    os.chdir('../')
    model = SingleLayerPerceptron().fit(X, y, len(labels))
    return model, labels


def gui(model, labels):
    text = input("Paste text to classify: ")
    text = prepare_text(text, is_path=False)
    print(labels[model.predict(text)[0]])

if __name__ == "__main__":
    model, labels = train()
    while True:
        gui(model, labels)
