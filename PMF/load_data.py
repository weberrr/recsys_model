import numpy as np
import random

def load_rating_data(file_path):
    data = []
    maxu = 0
    maxi = 0
    maxr = 0.0
    with open(file_path, "r") as f:
        for line in f:
            if line:
                triple = line.split("::")
                user, item, rating = int(triple[0]), int(
                    triple[1]), float(triple[2])
                data.append((user, item, rating))
                if user > maxu:
                    maxu = user
                if item > maxi:
                    maxi = item
                if rating > maxr:
                    maxr = rating
    print("Loading Success!\n Data info:\n \tUser num:{}\n \tItem num:{}\n \tData size:{}".format(
        maxu, maxi, len(data)))
    return np.array(data)


def split_rating_dat(data, size=0.2):
    train_data = []
    test_data = []
    for line in data:
        rand = random.random()
        if rand < size:
            test_data.append(line)
        else:
            train_data.append(line)
    return np.array(train_data), np.array(test_data)

if __name__ == "__main__":
    file_path = "PMF\data\ml-1m\\ratings.dat"
    load_rating_data(file_path)