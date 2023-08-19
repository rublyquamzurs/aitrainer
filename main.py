# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import csv
import numpy as np
from seek.entry import Entry


def get_train_data():
    t = []
    with open("ai/train/white.csv", "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for one in reader:
            t.append([one["domain"], one["label"]])
    with open("ai/train/black.csv", "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for one in reader:
            t.append([one["domain"], one["label"]])
    return np.array(t)


def main():
    data: np.array = get_train_data()
    np.random.shuffle(data)
    line = int(data.shape[0] / 10)
    entry = Entry(data[:-line], data[-line:])
    entry.run()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
