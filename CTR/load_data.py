import numpy as np
import random


def load_rating_data(file_path):
    data = []
    maxu = 5551
    maxi = 16980
    with open(file_path, "r") as f:
        uid = 0
        for line in f:
            if line:
                paper_idx = line.split(" ")
                for pid in paper_idx[1:]:
                    data.append((int(uid), int(pid), int(1)))
                negative_sample = 0
                while negative_sample < 10:
                    neg_pid = np.random.randint(0, maxi)
                    if neg_pid not in paper_idx:
                        data.append((int(uid), int(neg_pid), int(0)))
                        negative_sample += 1
            uid += 1
    print("Loading Success!\n Data info:\n \tUser num:{}\n \tItem num:{}\n \tData size:{}".format(
        maxu, maxi, len(data)))
    return np.array(data)


def load_paper_content(file_path):
    data = np.zeros((16980, 8000))
    with open(file_path, "r") as f:
        for line in f:
            if line:
                p_w_num = line.split(" ")
                data[int(p_w_num[0])][int(p_w_num[1])] = int(p_w_num[2])
        f.close()
    print("Loading Side Information Success!")
    return data


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


def get_users_items(train_set, test_set):
    num_user = int(
        max(np.amax(train_set[:, 0]), np.amax(test_set[:, 0])))+1  # user总数
    num_item = int(
        max(np.amax(train_set[:, 1]), np.amax(test_set[:, 1])))+1  # item总数
    users = [[] for _ in range(num_user)]
    items = [[] for _ in range(num_item)]
    for i in np.concatenate((train_set,test_set),axis=0):
        if int(i[2]) == 1:
            if int(i[1]) not in users[int(i[0])]:
                users[int(i[0])].append(int(i[1]))
            if int(i[0]) not in items[int(i[1])]:
                items[int(i[1])].append(int(i[0]))
    return users, items
