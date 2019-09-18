import numpy as np


def load_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for p in file:
            l = []
            line = p.strip().split()
            if int(line[0]) != 0:
                for x in line[1:]:
                    l.append(int(x))
            data.append(np.array(l, dtype=np.int32))
    return np.array(data)


def load_content_data(file_path):
    data = np.zeros((16980, 8000))
    with open(file_path, "r") as f:
        for line in f:
            if line:
                p_w_num = line.split(" ")
                data[int(p_w_num[0])][int(p_w_num[1])] = int(p_w_num[2])
        f.close()
    print("Loading Side Information Success!")
    return data


def split_rating_dat(data, size):
    num_user = 5551
    train_data = [[] for _ in range(num_user)]
    test_data = [[] for _ in range(num_user)]
    t = 0
    for line in data:
        num = line.strip().split(" ")
        temp = 0
        for n in num:
            if temp < size:
                train_data[t].append(n)
            else:
                test_data[t].append(n)
            temp += 1
        t += 1
    train_item = [[] for _ in range(num_user)]
    for n, i in enumerate(train_data):
        for k in i:
            train_item[k].append(n)
    return np.array(train_data), np.array(test_data), np.array(train_item)
