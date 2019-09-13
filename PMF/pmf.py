import numpy as np


class PMF():
    def __init__(self, num_feature=20, max_epoch=10, num_batch=50, batch_size=5000, _lambda=0.1, epsilon=0.9):
        self.num_feature = num_feature   # Number of latent features,
        self.max_epoch = max_epoch  # Number of epoch before stop,
        # Number of batches in each epoch (for SGD optimization),
        self.num_batch = num_batch
        # Number of training samples used in each batches (for SGD optimization)
        self.batch_size = batch_size
        self._lambda = _lambda  # L2 regularization,
        self.epsilon = epsilon  # learning rate,

        # for draw the performaince
        self.rmse_train = []
        self.rmse_test = []

    def fit(self, train_set=None, test_set=None):

        num_user = int(
            max(np.amax(train_set[:, 0]), np.amax(test_set[:, 0])))+1  # user总数
        num_item = int(
            max(np.amax(train_set[:, 1]), np.amax(test_set[:, 1])))+1  # item总数

        self.U = 0.1*np.random.randn(num_user, self.num_feature)
        self.U = np.array(self.U, dtype=np.float64)
        self.V = 0.1*np.random.randn(num_item, self.num_feature)
        self.V = np.array(self.V, dtype=np.float64)

        epoch = 0  # 迭代次数
        while epoch < self.max_epoch:
            epoch += 1
            shuffled_order = np.arange(train_set.shape[0])  # 根据记录创建等差array
            np.random.shuffle(shuffled_order)  # 将一个列表中的元素随机打乱

            for batch in range(self.num_batch):  # 每次使用的数据量
                batch_data = np.arange(self.batch_size*batch,
                                       self.batch_size*(batch+1))
                batch_idx = np.mod(batch_data, shuffled_order.shape[0])
                batch_users_id = np.array(
                    train_set[shuffled_order[batch_idx], 0], dtype="int32")
                batch_items_id = np.array(
                    train_set[shuffled_order[batch_idx], 1], dtype="int32")

                # Compute Error
                predicts = np.sum(np.multiply(
                    self.U[batch_users_id, :], self.V[batch_items_id, :]), axis=1)
                errors = predicts - train_set[shuffled_order[batch_idx], 2]
                
                # Compute Gradients
                U_grad = np.multiply(
                    errors[:, np.newaxis], self.V[batch_items_id, :])+self._lambda*self.U[batch_users_id, :]
                V_grad = np.multiply(
                    errors[:, np.newaxis], self.U[batch_users_id, :])+self._lambda*self.V[batch_items_id, :]

                # find same element to update
                U_2_update = np.zeros((num_user, self.num_feature))
                V_2_update = np.zeros((num_item, self.num_feature))
                for i in range(self.batch_size):
                    U_2_update[batch_users_id[i], :] = U_grad[i, :]
                    V_2_update[batch_items_id[i], :] = V_grad[i, :]

                # Update with epsilon
                self.U = self.U - self.epsilon * U_2_update
                self.V = self.V - self.epsilon * V_2_update

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
            print("Training RMSE:%f,Test RMSE:%f"%(train_loss, test_loss))

    def evaluate(self, predict_val, observe_val, size):
        error = predict_val - observe_val
        loss = np.sum(error**2)
        return np.sqrt(loss/size)

    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feature = parameters.get("num_feature", 20)
            self.max_epoch = parameters.get("max_epoch", 10)
            self.num_batch = parameters.get("num_batch", 50)
            self.batch_size = parameters.get("batch_size", 5000)
            self._lambda = parameters.get("_lambda", 0.1)
            self.epsilon = parameters.get("epsilon", 0.9)


if __name__ == '__main__':
    model = PMF()
