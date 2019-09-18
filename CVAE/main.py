import numpy as np 
import matplotlib.pyplot as plt
from load_data import load_data,load_content_data,split_rating_dat
from cvae import CVAE

P = 5
NUM_ITEM = 16980
ITEM_EMB = 8000

if __name__ == "__main__":
    data_dir = "CVAE\\data\\citeulike-a\\"
    train_users,test_users,train_items = split_rating_dat(load_data(data_dir + "users.dat"),P)
    side_info = load_content_data(data_dir + "papers.dat",NUM_ITEM,ITEM_EMB)
    
    model = CVAE(num_features=50, max_epoch=50 , max_iter=5, a=1, b=0.01, lambda_u=0.1, lambda_v=10, lambda_r=10,vae_pre_training="CVAE\\data\\vae_pre_training.pt")
    model.fit(train_users,test_users,train_items, side_info)

    # VAE 经过逐层预训练(单独用每一层的小网络训练)，得到预训练参数 vae_pre_training.pt 
    # 在citeulike-a上实验，在经过约 10 个epoch 后，recall@50 达到0.16，复现了论文中的实验结果数值


