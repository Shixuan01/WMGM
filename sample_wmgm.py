from __future__ import print_function
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader

import os
import random
import Attention_GAN
from torchvision import datasets, utils, transforms
from utils.dataset_wavelet import wavelet_inverse, wavelet_transform
import argparse
import cv2
from utils.args import add_dict_to_argparser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_argparser():
    defaults = dict(
        batch_size=128,
        image_size=128,
        n_channels=64,
        n_blocks=5,
        input_ch_num=6,
        output_ch_num=9,
        netD_ch_num=3,
        checkpoint=3,

        # data loading
        n_threads=8,
        crop_size=256,
        max_epoch=100,
        epoch_len=500,
        max_epochs=100,
        data_queue_len=10000,
        patch_per_tile=10,
        color_space="RGB",

        depth=2,
        wavelet_channel=3,
        data_path="raw_data\\valid",
        model_path="",
        save_path="output"
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    setup_seed(0)
    cont_config = create_argparser().parse_args()

    if not os.path.exists(cont_config.save_path):
        os.makedirs(cont_config.save_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netG = Attention_GAN.Generator(
        cont_config,
        in_channels=cont_config.input_ch_num,
        out_channels=cont_config.output_ch_num,
        padding=1,
        batch_norm=True,
        pooling_mode="maxpool",
    )

    netG_dict = torch.load(cont_config.model_path)
    netG.load_state_dict(netG_dict)
    netG.to(device)

    data_set = datasets.ImageFolder(root=cont_config.data_path, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    data_loader = DataLoader(
        dataset=data_set,
        num_workers=1,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    netG.eval()
    j = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            fnames = []
            img = batch[0].to(device)
            for i in range(img.shape[0]):
                fnames.append(data_set.imgs[j+i][0][-9:-4])

            resize = []
            img = img * 4.0
            for dp in reversed(range(cont_config.depth)):
                noise = torch.randn_like(img).to(device)
                output = netG(torch.cat((img, noise), dim=1))
                img = torch.cat((torch.cat((img, output[:, :3, :, :]), dim=3),
                                 torch.cat((output[:, 3:6, :, :], output[:, 6:9, :, :]), dim=3)), dim=2)
                img = wavelet_inverse(img, cont_config.wavelet_channel, cont_config.image_size // (2 ** dp), device)
            for i in range(img.shape[0]):
                utils.save_image(
                    img[i],
                    "%s/%s.png" % (cont_config.save_path, fnames[i])
                )
            j += img.shape[0]
