 # pylint: disable=no-member
 # pylint: disable=not-callable
import numpy as np
import scipy
import torch
from torch import optim
from vae import VAE


class CVAE():
    def __init__(self, num_features=50, max_epoch=50, max_iter=5, a=1, b=0.01, lambda_u=0.1, lambda_v=10, lambda_r=10, vae_pre_training=None):
        self.num_features = num_features
        self.max_epoch = max_epoch
        self.max_iter = max_iter
        self.a = a
        self.b = b
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_r = lambda_r
        self.vae_pre_training = vae_pre_training

    def initialize(self, train_users, train_items, item_side_info):
        self.num_users = len(train_users)
        self.num_items = len(train_items)
        self.U = 0.1*np.random.randn(self.num_users, self.num_features)
        self.V = 0.1*np.random.randn(self.num_items, self.num_features)
        self.V_theta = 0.1*np.random.randn(self.num_items, self.num_features)
        self.vae = VAE([item_side_info.shape[1], 200, 100], self.num_features)
        side_input = torch.tensor(item_side_info, dtype=torch.float)
        if self.vae_pre_training != None:
            self.vae.load_state_dict(torch.load(self.vae_pre_training))
            self.V_theta[:] = self.vae.encoder(
                side_input).clone().detach().numpy()
            self.V[:] = self.V_theta
        self.optimizer = optim.Adam(self.vae.parameters(
        ), lr=0.001, weight_decay=2e-4)  # weight_decay为L2正则化
        return side_input

    def fit(self, train_users, test_users, train_items, item_side_info):
        side_input = self.initialize(train_users, train_items, item_side_info)
        for epoch in range(self.max_epoch):
            loss, side_latent = self.e_step(side_input)
            self.V[:] = side_latent.clone().detach().numpy()
            recall = self.m_step(train_users, train_items, test_users)
            print("Epoch:{}, Loss:{}, Recall:{}".format(epoch, loss, recall))

    # fix U,V  update V_theta
    def e_step(self, side_input):
        loss = 0.
        for it in range(self.max_iter):
            self.optimizer.zero_grad()
            side_latent = self.vae.encoder(side_input)
            side_output = self.reg_tensor(self.vae.decoder())
            gen_loss = -torch.mean((side_input*torch.log(side_output) +
                                    (1-side_output)*torch.log(side_output)).sum(dim=1))
            latent_loss = self.vae.latent_loss()
            v_loss = self.lambda_r * \
                torch.mean(
                    ((side_latent-torch.tensor(self.V, dtype=torch.float))**2).sum(dim=1))
            loss = gen_loss + latent_loss + v_loss
            loss.backward()
            self.optimizer.step()
            print("E_Step：Iter:{}, Loss:{:.5f}, gen_loss:{:0.5f}, latent_loss:{:0.5f}, v_loss:{:0.5f}"
                  .format(it, loss, gen_loss, latent_loss, v_loss))
        return loss, side_latent

    # fix V_theta  update U,V
    def m_step(self, train_users, train_items, test_users):
        for it in range(self.max_iter):
            # update U
            items_ids = np.array([len(x) for x in train_items]) > 0
            v = self.V[items_ids]
            vTv = np.dot(v.T, v)*self.b
            for i in range(self.num_users):
                ui_items = train_users[i]
                if len(ui_items) > 0:
                    fs_part = vTv + \
                        np.dot(self.V[ui_items, :].T,
                               self.V[ui_items, :])*(self.a-self.b)
                    fs_part += self.lambda_u*np.eye(self.num_features)
                    sec_part = np.sum(self.V[ui_items, :], axis=0)*self.a
                    try:
                        self.U[i, :] = scipy.linalg.solve(fs_part, sec_part)
                    except AttributeError:
                        # if module 'scipy' has no attribute 'linalg'
                        self.U[i, :] = np.dot(np.mat(fs_part).I, sec_part)
            # update V
            users_ids = np.array([len(x) for x in train_users]) > 0
            u = self.U[users_ids]
            uTu = np.dot(u.T, u)*self.b
            for j in range(self.num_items):
                vj_users = train_items[j]
                if len(vj_users) > 0:
                    fs_part = uTu + \
                        np.dot(self.U[vj_users, :].T,
                               self.U[vj_users, :])*(self.a-self.b)
                    fs_part += self.lambda_v*np.eye(self.num_features)
                    sec_part = np.sum(
                        self.U[vj_users, :], axis=0)*self.a + self.lambda_v * self.V_theta[j, :]
                else:
                    fs_part = uTu + self.lambda_v*np.eye(self.num_features)
                    sec_part = self.lambda_v * self.V_theta[j, :]
                try:
                    self.V[j, :] = scipy.linalg.solve(fs_part, sec_part)
                except AttributeError:
                    # if module 'scipy' has no attribute 'linalg'
                    self.V[j, :] = np.dot(np.mat(fs_part).I, sec_part)
            recall = self.evalute_recall(train_users, test_users, [50,100,150])
            print("M_Step：Iter:{}, Recall@50:{:.5f}, Recall@100:{:.5f},Recall@150:{:.5f}"
                  .format(it, recall[0], recall[1], recall[2]))
        return recall[0]

    def reg_tensor(self, ts):
        return torch.max(torch.sigmoid(ts), torch.tensor(1e-10, dtype=torch.float))

    def evalute_recall(self, train_users, test_users, recall_M):
        res = []
        score = np.dot(self.U, self.V.T)
        ind_rec = np.argsort(score, axis=1)[:, ::-1]
        for m in recall_M:
            recalls = []
            for i in range(self.num_users):
                if len(test_users[i]) > 0:
                    m_rec = []
                    recall = 0.
                    for j in ind_rec[i]:
                        if j not in train_users[i]:
                            m_rec.append(j)
                            if j in test_users[i]:
                                recall += 1.
                        if len(m_rec) == m:
                            break
                    recalls.append(recall/len(test_users[i]))
            res.append(np.mean(recalls))
        return res

    def save_model(self, file_path):
        pass

    def load_model(self, file_path):
        pass
