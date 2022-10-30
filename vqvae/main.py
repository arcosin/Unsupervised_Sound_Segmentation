
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Normalize, Compose, ToTensor
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pickle
import torch
from vq_vae import VQVAE

def read_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
vae = VQVAE(1, 28, 100, [16, 64, 128]).to(DEVICE)
transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

#image_path = "C:\\Users\\elico\\Documents\\Unsupervised_Sound_Segmentation\\dataset\\processed\\ZOOM0009\\spectrogram\\"
train_data = read_pkl('../data/train_tensor-001.pkl')
test_data = read_pkl('../data/test_tensor.pkl')
#X = []
#tensor_transform = transforms.ToTensor()
"""
for file in os.listdir(image_path):
    fname = os.fsdecode(file)
    png = Image.open(image_path + fname)
    png.load()
    newImg = Image.new("RGB", png.size, (255, 255, 255))
    newImg.paste(png, mask=png.split()[3])
    newImg.save(image_path + 'temp.png', 'PNG')
    X.append(tensor_transform(Image.open(image_path + 'temp.png')))
os.remove(image_path + 'temp.png')
"""
"""
for file in os.listdir(image_path):
    fname = os.fsdecode(file)
    img = Image.open(image_path + fname).convert('L')
    X.append(tensor_transform(img))

class Custom_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example = self.dataset[index]
        return np.array(example)

    def __len__(self):
        return len(self.dataset)


X = Custom_Dataset(X)
"""
#dataloader = torch.utils.data.DataLoader(X, batch_size=512)
train_ds = TensorDataset(train_data.to(DEVICE))
test_ds = TensorDataset(test_data.to(DEVICE))
train_dataloader = DataLoader(train_ds, batch_size=512)
test_dataloader = DataLoader(train_ds, batch_size=512)

optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3)

# Train
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

# vae.load_state_dict(torch.load('./vae.pt'))
torch.save(vae, "./weights")
#vae = torch.load("./weights")
vae.eval()

"""
X_test = []
tensor_transform = transforms.ToTensor()
for file in os.listdir(image_path):
    fname = os.fsdecode(file)
    img = Image.open(image_path + fname).convert('L')
    X_test.append(tensor_transform(img))
    """
# Reconstruction
#dataloader_test = torch.utils.data.DataLoader(X, batch_size=512)
for x in test_dataloader:
    x = x.to(DEVICE).float()
    reconstruct_x = vae.generate(x)
    new_x = torch.cat([x, reconstruct_x.detach()], dim=0)
    grid_pics = make_grid(new_x.to('cpu'), 8)
    plt.imshow(grid_pics.permute(1, 2, 0))
    break
