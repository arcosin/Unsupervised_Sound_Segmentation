
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Normalize, Compose, ToTensor
import numpy as np
import pickle
import torch
from vq_vae import VQVAE


def read_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data


DEVICE = 'cuda:0' if torch.cuda.is_available() else (
    'mps' if torch.backends.mps.is_available() else 'cpu')
vae = VQVAE(1, 28, 100, [16, 64, 128]).to(DEVICE)
transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

train_data = read_pkl("train_spectograms.p")
test_data = read_pkl('test_spectograms.p')

train_ds = TensorDataset(train_data.to(DEVICE))
test_ds = TensorDataset(test_data.to(DEVICE))
train_dataloader = DataLoader(train_ds, batch_size=64)
test_dataloader = DataLoader(test_ds, batch_size=64)

optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3)

# Train
"""
tqdm_bar = tqdm(range(5))
for ep in tqdm_bar:
    for i, x in enumerate(train_dataloader):
        x = x[0].to(DEVICE).float()
        x = x[:, None, :, :]
        with torch.autocast(DEVICE):
            recon, input, vq_loss = vae(x)
            loss = vae.loss_function(recon, input, vq_loss)
        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            tqdm_bar.set_description('loss: {}'.format(loss['loss']))
"""
# vae.load_state_dict(torch.load('./vae.pt'))
#torch.save(vae, "./weights")

# for test
vae = torch.load("./weights")
vae.eval()

# Reconstruction
#dataloader_test = torch.utils.data.DataLoader(X, batch_size=512)
for x in test_dataloader:
    x = x[0].to(DEVICE).float()
    x = x[:, None, :, :]
    reconstruct_x = vae.generate(x)
    #new_x = torch.cat([x, reconstruct_x.detach()], dim=0)
    i = 0
    x_orig = x[i]
    x_new = reconstruct_x[i]
    #grid_pics = make_grid(new_x.to('cpu'), 8)
    #plt.imshow(grid_pics.permute(1, 2, 0))
    plt.imshow(x_orig.permute(1,2,0))
    plt.axis('off')
    plt.savefig(f'../VQVAE/sounds/x{i}_orig.png', bbox_inches='tight', pad_inches = 0)
    plt.imshow(x_new.permute(1, 2, 0).detach().numpy())
    plt.axis('off')
    plt.savefig(f'../VQVAE/sounds/x{i}_new.png', bbox_inches='tight', pad_inches = 0)

    with open(f'../VQVAE/sounds/x{i}_orig.npy', 'wb') as f:
        np.save(f, np.squeeze(np.array(x_orig)))
    with open(f'../VQVAE/sounds/x{i}_new.npy', 'wb') as g:
        np.save(g, np.squeeze(x_new.detach().numpy()))
    break
