from load_data import load_rating_data,split_rating_dat,load_paper_content
from ctr import CTR
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ratings_path = r"CTR\data\citeulike-a\users.dat"
    paper_content_path = r"CTR\data\citeulike-a\papers.dat"
    ctr = CTR()
    ctr.set_params({"num_feature": 30, "max_iter": 200, "a": 1, "b":0.01, 
                    "_lambda_u": 0.1, "_lambda_v": 10})
    ratings_data = load_rating_data(ratings_path)
    side_data = load_paper_content(paper_content_path)
    train, test = split_rating_dat(ratings_data, size=0.2)
    ctr.fit(train,test,side_data)

    # Check performance by plotting train and test errors
    plt.plot(range(ctr.max_iter), ctr.rmse_train, marker='o', label='Training Data')
    plt.plot(range(ctr.max_iter), ctr.rmse_test, marker='v', label='Test Data')
    plt.title('The Citeulike Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()