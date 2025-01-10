import h5py
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import Dataset


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset, y, i, datasetname):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(1000):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()
            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            # kmeans = KMeans(n_clusters=6, n_init=20).fit(z.data.cpu().numpy())
            # eva(y, kmeans.labels_, epoch)
        torch.save(model.state_dict(), '{}{}.pkl'.format(datasetname, i))




model1 = AE(
        n_enc_1=128,
        n_enc_2=256,
        n_enc_3=512,
        n_dec_1=512,
        n_dec_2=256,
        n_dec_3=128,
        n_input=3183,      #第一个视角的维度
        n_z=10,).cuda()


model2 = AE(
        n_enc_1=128,
        n_enc_2=256,
        n_enc_3=512,
        n_dec_1=512,
        n_dec_2=256,
        n_dec_3=128,
        n_input=3203,       #第二个视角的维度
        n_z=10,).cuda()





dataset = 'BBCSport'


if dataset == 'BBCSport':
    # 两个视角维度：3183 3203
    x1 = scio.loadmat('data/{}.mat'.format(dataset))['X1']
    x2 = scio.loadmat('data/{}.mat'.format(dataset))['X2']
    y = scio.loadmat('data/{}.mat'.format(dataset))['gt']
    x1 = np.transpose(x1)
    x2 = np.transpose(x2)
    y = np.squeeze(y)



for i in [1,2]:
    print('-----------------------------------------------------{}  view{}'.format(dataset, i))
    if i==1:
        data1 = LoadDataset(x1)
        pretrain_ae(model1, data1, y, i, dataset)
    if i==2:
        data2 = LoadDataset(x2)
        pretrain_ae(model2, data2, y, i, dataset)


