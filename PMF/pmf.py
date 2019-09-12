import numpy as np

def get_train_data(file_path):
    data = []
    u = 0
    i = 0
    maxr = 0.0
    with open(file_path, "r") as f:
        for line in f:
            if line:
                triple = line.split("::")
                user = int(triple[0])
                item = int(triple[1])
                score = float(triple[2])
                data.append((user, item, score))
                if user > u:
                    u = user
                if item > i:
                    i = item
                if score > maxr:
                    maxr = score
    print("Loading Success!\n Data info:\n \tUser num:{}\n \tItem num:{}\n \tData size:{}".format(
        u, i, len(data)))
    R = np.zeros([u+1, i+1], dtype=np.float32)
    for i in data:
        user = i[0]
        item = i[1]
        rating = i[2]
        R[user][item] = rating
    print(R.shape)
    return R

if __name__ == "__main__":
    file_path = "PMF\data\ml-1m\\ratings.dat"
    get_train_data(file_path)
