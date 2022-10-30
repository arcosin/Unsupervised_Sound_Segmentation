
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.cuda.amp import autocast_mode
from tqdm import tqdm
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose, ToTensor

import torch

from vq_vae import VQVAE


vae = VQVAE(1, 28, 100, [16, 64, 128]).to('cuda')
transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])
dataloader = DataLoader(MNIST(root='./', download=True,
                        transform=transform), batch_size=512)
optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3)

# Train
tqdm_bar = tqdm(range(5))
for ep in tqdm_bar:
    for i, (x, _) in enumerate(dataloader):
        x = x.to('cuda').float()
        with torch.autocast('cuda'):
            recon, input, vq_loss = vae(x)
            loss = vae.loss_function(recon, input, vq_loss)
        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            tqdm_bar.set_description('loss: {}'.format(loss['loss']))

# Reconstruction
dataloader_test = DataLoader(MNIST(
    root='./', download=True, transform=transform, train=False), batch_size=16, shuffle=True)
# vae.load_state_dict(torch.load('./vae.pt'))
vae.eval()
for x, _ in dataloader_test:
    x = x.to('cuda').float()
    reconstruct_x = vae.generate(x)
    new_x = torch.cat([x, reconstruct_x.detach()], dim=0)
    grid_pics = make_grid(new_x.to('cpu'), 8)
    plt.imshow(grid_pics.permute(1, 2, 0))
    break
