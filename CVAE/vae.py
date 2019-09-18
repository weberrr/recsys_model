import torch
from torch import nn,optim

# VAE need to 

class VAE(nn.Module):
    def __init__(self, dims, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1], bias=True)
        self.fc2 = nn.Linear(dims[1], dims[2], bias=True)
        self._enc_mu = nn.Linear(100, latent_dim, bias=True)
        self._enc_log_sigma = nn.Linear(dims[2], latent_dim, bias=True)
        self.z_fc = nn.Linear(50, dims[2], bias=True)
        self.fc3 = nn.Linear(dims[2], dims[1], bias=True)
        self.fc4 = nn.Linear(dims[1], dims[0], bias=True)

    def _sample_latent(self, x):
        mu = self._enc_mu(x)
        log_sigma = self._enc_log_sigma(x)
        sigma = torch.exp(log_sigma)
        std_z = torch.randn(sigma.size(), dtype=torch.float,
                            requires_grad=False)
        self.z_mean = mu
        min_num = torch.tensor(1e-10, dtype=torch.float)
        self.z_std = torch.max(min_num, sigma)
        return mu + sigma*std_z

    def encoder(self, x):
        a1 = torch.sigmoid(self.fc1(x))
        a2 = torch.sigmoid(self.fc2(a1))
        # notice that, use z_mean as latent_vector to output
        self.z = self._sample_latent(a2)
        return self.z_mean

    def decoder(self):
        a3 = self.z_fc(self.z)
        a4 = torch.sigmoid(self.fc3(a3))
        y = torch.sigmoid(self.fc4(a4))
        return y

    def forward(self, x):
        self.encoder(x)
        return self.decoder()

    # KL loss for vae
    def latent_loss(self):
        mean_sq = self.z_mean * self.z_mean
        stddev_sq = self.z_std * self.z_std
        return 0.5 *torch.mean((mean_sq+stddev_sq-torch.log(stddev_sq)-1).sum(dim=1))


if __name__ == "__main__":
    vae = VAE([50,20,10],5)

