import torch.nn.parallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, utils, transforms
from utils.dataset_wavelet import wavelet_transform
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="input path")
parser.add_argument("--output_dir", type=str, required=True, help="output path")
args = parser.parse_args()

train_set = datasets.ImageFolder(root=args.input_dir, transform=transforms.Compose([
    transforms.ToTensor(),
]))


train_data_loader = DataLoader(
    dataset=train_set,
    num_workers=0,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)
#
all_data = []
for i, batch in enumerate(tqdm(train_data_loader), 1):
    img = batch[0]
    for j in range(2):
        img,_ = wavelet_transform(img,3,128//(j+1),"cpu")
    all_data.append(img)
all_data = torch.cat(all_data,dim=0)
max = all_data.max()
min = all_data.min()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    print(f"'{args.output_dir}' has been created.")
else:
    print(f"'{args.output_dir}' already exists.")


for i in tqdm(range(all_data.shape[0])):
    img = all_data[i]
    img = (img-min)/(max-min)
    utils.save_image(img,args.output_dir+"/"+str(i)+".png")

