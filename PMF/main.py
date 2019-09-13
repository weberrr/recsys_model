from load_data import load_rating_data,split_rating_dat
from pmf import PMF
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = "PMF\data\ml-1m\\ratings.dat"
    pmf = PMF()
    pmf.set_params({"num_feature": 10, "max_epoch": 50, "num_batch": 50,
                    "batch_size": 1000, "epsilon": 0.01, "_lambda": 0.1})
    ratings_data = load_rating_data(file_path)
    train, test = split_rating_dat(ratings_data, size=0.2)
    pmf.fit(train,test)

    # Check performance by plotting train and test errors
    plt.plot(range(pmf.max_epoch), pmf.rmse_train, marker='o', label='Training Data')
    plt.plot(range(pmf.max_epoch), pmf.rmse_test, marker='v', label='Test Data')
    plt.title('The MovieLens Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()