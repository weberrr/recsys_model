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


def load_content_data(file_path,item_num,item_dim):
    data = np.zeros((item_num, item_dim))
    with open(file_path, "r") as f:
        for line in f:
            if line:
                p_w_num = line.split(" ")
                data[int(p_w_num[0])][int(p_w_num[1])] = int(p_w_num[2])
        f.close()
    # divide the max count for each paper
    res = []
    for i in data:
        max_count = max(i)
        if max_count != 0:
            res.append([k/max_count for k in i])
        else:
            res.append(i)
    print("Loading Side Information Success!")
    return np.array(res)


def split_rating_dat(data, size):
    num_user = len(data)
    num_item = 0
    train_data = [[] for _ in range(num_user)]
    test_data = [[] for _ in range(num_user)]
    idx = 0
    for line in data:
        num_item = max(max(line),num_item)
        if len(line) <= size:
            train_data[idx] = line
        else:
            np.random.shuffle(line)
            train_data[idx] = line[:size]
            test_data[idx] = line[size:]
        idx += 1
    train_item = [[] for _ in range(num_item+1)]
    for n, i in enumerate(train_data):
        for k in i:
            train_item[k].append(n)
    print("Loading Intraction Data Success!")
    return np.array(train_data), np.array(test_data), np.array(train_item)


if __name__ == '__main__':
    data_dir = "CVAE\\data\\citeulike-a\\"
    train_users,test_users,train_items = split_rating_dat(load_data(data_dir + "users.dat"),5)
    side_info = load_content_data(data_dir + "papers.dat",16980,8000)
    print(train_users[:5])
    
    
