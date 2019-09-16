import numpy as np
import scipy
from load_data import get_users_items
from sklearn.decomposition import LatentDirichletAllocation


class CTR():
    def __init__(self, num_feature=20, max_iter=20, a=1, b=0.01, _lambda_u=0.1,_lambda_v=10):
        self.num_feature = num_feature   # Number of latent features,
        self.max_iter = max_iter  # Number of epoch before stop,
        self.a = a  # when ui and vj have interaction, C_ij = a ,
        self.b = b  # when not have, C_ij = b ,
        # Number of batches in each epoch (for SGD optimization),
        self._lambda_u = _lambda_u  # L2 regularization for u,
        self._lambda_v = _lambda_v  # L2 regularization for v,

        # for draw the performaince
        self.rmse_train = []
        self.rmse_test = []

    def fit(self, train_set=None, test_set=None, side_info=None):

        num_user = int(max(np.amax(train_set[:, 0]), np.amax(test_set[:, 0])))+1  # user总数
        num_item = int(max(np.amax(train_set[:, 1]), np.amax(test_set[:, 1])))+1  # item总数
        users, items = get_users_items(train_set, test_set)

        # initialize
        self.U = np.array(0.1*np.random.randn(num_user,self.num_feature), dtype=np.float64)
        lda = LatentDirichletAllocation(n_components=self.num_feature,learning_method='batch')
        self.V_theta = np.array(lda.fit_transform(side_info), dtype=np.float64)
        self.V_epsilon = np.array(0.1*np.random.randn(num_item, self.num_feature), dtype=np.float64)
        self.V = self.V_theta + self.V_epsilon

        for _ in range(self.max_iter):
            # Compute Gradients(Coordinate descent) & update
            # update U
            all_item_ids = np.array([len(x) for x in items]) > 0
            v = self.V[all_item_ids, :]
            vTv = np.dot(v.T, v)
            b_part = vTv*self.b + np.eye(self.num_feature)*self._lambda_u  # vTv*b 防止矩阵全部拟合为1
            for i in range(num_user):
                item_ids = users[i]
                if len(item_ids) > 0:
                    a_part = np.dot(self.V[item_ids, :].T, self.V[item_ids, :])*(self.a-self.b)
                    first_part = b_part + a_part
                    second_part = self.a * np.sum(self.V[item_ids, :], axis=0)
                    # self.U[i, :] = np.dot(np.mat(first_part).I,second_part)
                    self.U[i, :] = scipy.linalg.solve(first_part, second_part)
            # update V
            all_user_ids = np.array([len(x) for x in users]) > 0
            u = self.U[all_user_ids, :]
            uTu = np.dot(u.T, u)
            b_part = uTu*self.b + np.eye(self.num_feature)*self._lambda_v
            for j in range(num_item):
                user_ids = items[j]
                if len(user_ids) > 0:
                    a_part = np.dot(self.U[user_ids, :].T, self.U[user_ids, :])*(self.a-self.b)
                    first_part = b_part + a_part
                    theta_part = self._lambda_v*self.V_theta[j, :]
                    second_part = self.a * np.sum(self.U[user_ids, :], axis=0) + theta_part  # 多 theta 项
                    # self.V[j, :] = np.dot(np.mat(first_part).I,second_part)
                    self.V[j, :] = scipy.linalg.solve(first_part, second_part)
                else:
                    theta_part = self._lambda_v*self.V_theta[j, :]
                    # self.V[j, :] = np.dot(np.mat(b_part).I,second_part)
                    self.V[j, :] = scipy.linalg.solve(b_part, theta_part)
            # train loss
            predict_val = np.sum(np.multiply(self.U[np.array(
                train_set[:, 0], dtype="int32")], self.V[np.array(train_set[:, 1], dtype="int32")]), axis=1)
            observe_val = train_set[:, 2]
            train_loss = self.evaluate(
                predict_val, observe_val, train_set.shape[0])
            self.rmse_train.append(train_loss)
            # val loss
            test_predict_val = np.sum(np.multiply(self.U[np.array(
                test_set[:, 0], dtype="int32")], self.V[np.array(test_set[:, 1], dtype="int32")]), axis=1)
            test_observe_val = test_set[:, 2]
            test_loss = self.evaluate(
                test_predict_val, test_observe_val, test_set.shape[0])
            self.rmse_test.append(test_loss)
            print("Training RMSE:%f,Test RMSE:%f" % (train_loss, test_loss))

    def evaluate(self, predict_val, observe_val, size):
        error = predict_val - observe_val
        loss = np.sum(error**2)
        return np.sqrt(loss/size)

    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feature = parameters.get("num_feature", 20)
            self.max_iter = parameters.get("max_iter", 30)
            self.a = parameters.get("a", 1)
            self.b = parameters.get("b", 0.01)
            self._lambda_u = parameters.get("_lambda_u", 0.1)
            self._lambda_v = parameters.get("_lambda_v", 10)

if __name__ == '__main__':
    model = CTR()
