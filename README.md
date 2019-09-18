# recsys model
Implementation for classic RecommenderSystem models with Pytorch

---

## Model List
| Model|	Paper |
| :------: | :------ |
| PMF(Probabilistic Matrix Factorization)| [NIPS 2007] [Probabilistic matrix factorization](http://xueshu.baidu.com/usercenter/paper/show?paperid=89ea382da660386bc91de376ff9e1087&site=xueshu_se)|
| CTR(Collaborative Topic Regression)| [KDD 2011] [Collaborative Topic Modeling for Recommending Scientific Articles](http://www.cs.columbia.edu/~blei/papers/WangBlei2011.pdf)|
| CVAE(Collaborative Variantional AutoEncoder)| [KDD 2017] [Collaborative Variational Autoencoder for Recommender Systems](https://dl.acm.org/citation.cfm?id=3098077)|

---

## Project Structure
```
project(e.g PMF)
│
│───main.py : to run the model example with data
│
│───model.py  : the model's implementation
│
│───load_data.py  : load data from '/data/xx'
│
└─── /data
│     │
│     └─── /data_name
│               │   
│               └─── data1.txt
│               │   
│               └─── data2.dat
│               │   
│               └─── README.md
│               │   
│               │   ...
```

