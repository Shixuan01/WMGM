from __future__ import print_function
import os
from time import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import pytorch_ssim
import Attention_GAN
from tqdm import tqdm
from torchvision import datasets, utils, transforms
from utils.dataset_wavelet import wavelet_transform, wavelet_inverse
import argparse
from utils.args import add_dict_to_argparser
import numpy as np
import random


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

        lamda=20,  # mse weight
        nu=0.5,  # ssim weight
        alpha=0.1,  # dis weight
        lr_G=1e-4,
        lr_D=1e-5,

        # data loading
        n_threads=8,
        crop_size=256,
        max_epoch=100,
        epoch_len=500,
        max_epochs=100,
        data_queue_len=10000,
        patch_per_tile=10,
        color_space="RGB",

        train_images_dir="./celeba_data/train",
        valid_images_dir="./celeba_data/valid",

        depth=2,
        wavelet_channel=3,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    setup_seed(0)
    train_config = create_argparser().parse_args()
    valid_config = create_argparser().parse_args()
    cont_flag = False
    retr_epoch_id = 99

    total_iters = train_config.max_epoch
    Generator_iters = 4  # every n epoches G is trained, D will be trained once

    im_save_path = "Validation_images_norm"
    if cont_flag:
        model_save_path = r'./checkpoint'
        summary_path = r'./summary'
    else:
        datetime_str = datetime.now().strftime("%Y%m%d-%H%M")
        model_save_path = (
                "model/"
                + datetime_str
                + im_save_path
                + "-lr_G={:.0e}-lr_D={:.0e}_norm".format(train_config.lr_G, train_config.lr_D)
        )
        summary_path = (
                "log/"
                + datetime_str
                + im_save_path
                + "-lr_G={:.0e}-lr_D={:.0e}_norm".format(train_config.lr_G, train_config.lr_D)
        )

        if not os.path.exists(im_save_path):
            os.makedirs(im_save_path)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)

    print("===> Building model")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # build generator and discriminator
    netG = Attention_GAN.Generator(
        train_config,
        in_channels=train_config.input_ch_num,
        out_channels=train_config.output_ch_num,
        padding=1,
        batch_norm=True,
        pooling_mode="maxpool",
    )

    netD = Attention_GAN.Discriminator(
        train_config, in_channels=train_config.netD_ch_num, batch_norm=True
    )

    netG.to(device)
    netD.to(device)

    if cont_flag:
        netG_dict = torch.load(
            model_save_path + "/netG_epoch_{}.pth".format(retr_epoch_id)
        )
        netG.load_state_dict(netG_dict)

        netD_dict = torch.load(
            model_save_path + "/netD_epoch_{}.pth".format(retr_epoch_id)
        )
        netD.load_state_dict(netD_dict)

    # define training and validation sets
    train_set = datasets.ImageFolder(root=train_config.train_images_dir, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    valid_set = datasets.ImageFolder(root=valid_config.valid_images_dir, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    train_data_loader = DataLoader(
        dataset=train_set,
        num_workers=train_config.n_threads,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    valid_data_loader = DataLoader(
        dataset=valid_set,
        num_workers=valid_config.n_threads,
        batch_size=valid_config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    # print(len(train_set), len(valid_set))

    writer = SummaryWriter(summary_path)

    # loss functions
    bce_loss_fun = nn.BCEWithLogitsLoss(size_average=True).to(device)

    mse_loss_fun = nn.MSELoss(size_average=True).to(device)

    l1_loss_fun = nn.L1Loss(size_average=True).to(device)

    ssim_loss_fun = pytorch_ssim.SSIM(size_average=True).to(device)

    real_label = 1
    fake_label = 0
    label_real = torch.full(
        (train_config.batch_size, 1),
        real_label,
        dtype=torch.float,
        device=device,
        requires_grad=False,
    )
    label_fake = torch.full(
        (train_config.batch_size, 1),
        fake_label,
        dtype=torch.float,
        device=device,
        requires_grad=False,
    )

    # setup optimizer
    optimizerG = optim.AdamW(
        netG.parameters(), lr=train_config.lr_G, betas=(0.9, 0.999)
    )
    optimizerD = optim.AdamW(
        netD.parameters(), lr=train_config.lr_D, betas=(0.9, 0.999)
    )

    print("===> Training Start")
    valid_log = open("valid_log.txt", "w")

    if cont_flag:
        startEpoch = retr_epoch_id
    else:
        startEpoch = 0

    niter = total_iters
    flag = 0
    for epoch in range(startEpoch, niter, 1):
        start_time = time()
        # train
        run_loss_G = 0
        run_loss_G_l1 = 0
        run_loss_G_ssim = 0
        run_loss_D_real = 0
        run_loss_D_fake = 0

        run_loss_G_list = []
        run_loss_G_l1_list = []
        run_loss_G_ssim_list = []

        run_loss_D_real_list = []
        run_loss_D_fake_list = []
        run_loss_D_list = []

        counter = 0
        netG.train()
        netD.train()
        for dp in range(train_config.depth):

            for i, batch in enumerate(tqdm(train_data_loader), 1):

                img = batch[0].to(device)
                for d in range(dp + 1):
                    img, target = wavelet_transform(img, train_config.wavelet_channel,
                                                    train_config.image_size // (2 ** d), device)
                noise = torch.randn_like(img).to(device)
                input = torch.cat((img, noise), dim=1)

                target = wavelet_inverse(
                    torch.cat((
                        torch.cat((img, target[:, :3, :, :]), dim=3),
                        torch.cat((target[:, 3:6, :, :], target[:, 6:9, :, :]), dim=3)
                    ), dim=2),
                    train_config.wavelet_channel,
                    train_config.image_size // (2 ** d),
                    device
                )
                ############################
                # (1) Update G network: maximize log(D(G(z)))
                ###########################
                fake = netG(input)
                fake = wavelet_inverse(
                    torch.cat((
                        torch.cat((img, fake[:, :3, :, :]), dim=3),
                        torch.cat((fake[:, 3:6, :, :], fake[:, 6:9, :, :]), dim=3)
                    ), dim=2),
                    train_config.wavelet_channel,
                    train_config.image_size // (2 ** d),
                    device
                )

                netG.zero_grad()
                # G_dis_loss = bce_loss_fun(netD(fake), label_real)
                G_dis_loss = -torch.mean(F.sigmoid(netD(fake)))

                G_ssim_loss = 1 - ssim_loss_fun((fake + 1.0) / 2.0, (target + 1.0) / 2.0)

                G_mse_loss = mse_loss_fun(fake, target)

                errG = (
                        train_config.alpha * G_dis_loss
                        + train_config.lamda * G_mse_loss
                        + train_config.nu * G_ssim_loss
                )

                clip_grad_norm_(netG.parameters(), 0.5)
                errG.backward()
                optimizerG.step()

                ############################
                # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                indie_G = Generator_iters
                if i % indie_G == 0:
                    netD.zero_grad()
                    D_fake_loss = torch.mean(F.sigmoid(netD(fake.detach())))
                    D_real_loss = -torch.mean(F.sigmoid(netD(target)))
                    errD = (D_fake_loss + D_real_loss) * 0.5
                    clip_grad_norm_(netD.parameters(), 0.5)
                    errD.backward()
                    optimizerD.step()

                if i % indie_G == 0:
                    counter = counter + 1
                    run_loss_G = run_loss_G + errG.item()
                    run_loss_G_l1 = run_loss_G_l1 + train_config.lamda * G_mse_loss.item()
                    run_loss_G_ssim = run_loss_G_ssim + train_config.nu * G_ssim_loss.item()

                    run_loss_D_real = run_loss_D_real + D_real_loss.item()
                    run_loss_D_fake = run_loss_D_fake + D_fake_loss.item()

            run_loss_G_list.append(run_loss_G / counter)
            run_loss_G_l1_list.append(run_loss_G_l1 / counter)
            run_loss_G_ssim_list.append(run_loss_G_ssim / counter)

            run_loss_D_real_list.append(run_loss_D_real / counter)
            run_loss_D_fake_list.append(run_loss_D_fake / counter)
            run_loss_D_list.append((run_loss_D_real + run_loss_D_fake) * 0.5)

            run_loss_G = 0
            run_loss_G_l1 = 0
            run_loss_G_ssim = 0
            run_loss_D_real = 0
            run_loss_D_fake = 0
            counter = 0

        # valid
        val_loss_G = 0
        val_loss_G_l1 = 0
        val_loss_G_ssim = 0
        val_loss_D_real = 0
        val_loss_D_fake = 0

        val_loss_G_list = []
        val_loss_G_l1_list = []
        val_loss_G_ssim_list = []
        val_loss_D_real_list = []
        val_loss_D_fake_list = []
        valid_loss_D_list = []

        counter = 0
        netG.eval()
        netD.eval()

        with torch.no_grad():
            for dp in range(train_config.depth):
                for i, batch in enumerate(tqdm(valid_data_loader), 1):
                    img = batch[0].to(device)
                    for d in range(dp + 1):
                        img, target = wavelet_transform(img, train_config.wavelet_channel,
                                                        train_config.image_size // (2 ** d), device)

                    noise = torch.randn_like(img).to(device)
                    val_input = torch.cat((img, noise), dim=1)

                    val_target = wavelet_inverse(
                        torch.cat((
                            torch.cat((img, target[:, :3, :, :]), dim=3),
                            torch.cat((target[:, 3:6, :, :], target[:, 6:9, :, :]), dim=3)
                        ), dim=2),
                        train_config.wavelet_channel,
                        train_config.image_size // (2 ** d),
                        device
                    )
                    val_fake = netG(val_input)
                    val_fake = wavelet_inverse(
                        torch.cat((
                            torch.cat((img, val_fake[:, :3, :, :]), dim=3),
                            torch.cat((val_fake[:, 3:6, :, :], val_fake[:, 6:9, :, :]), dim=3)
                        ), dim=2),
                        train_config.wavelet_channel,
                        train_config.image_size // (2 ** d),
                        device
                    )
                    netG.zero_grad()
                    G_dis_loss = -torch.mean(F.sigmoid(netD(val_fake)))
                    # G_dis_loss = bce_loss_fun(netD(val_fake), label_real)
                    G_mse_loss = l1_loss_fun(val_fake, val_target)
                    G_ssim_loss = 1 - ssim_loss_fun((val_fake + 1.0) / 2.0, (val_target + 1.0) / 2.0)
                    errG = (
                            train_config.alpha * G_dis_loss
                            + train_config.lamda * G_mse_loss
                            + train_config.nu * G_ssim_loss
                    )

                    D_fake_loss = torch.mean(F.sigmoid(netD(val_fake)))
                    D_real_loss = -torch.mean(F.sigmoid(netD(val_target)))
                    # D_fake_loss = bce_loss_fun(netD(fake.detach()), label_fake)
                    # D_real_loss = bce_loss_fun(netD(val_target), label_real)
                    errD = (D_fake_loss + D_real_loss) * 0.5

                    counter = counter + 1
                    val_loss_G = val_loss_G + errG.item()
                    val_loss_G_l1 = val_loss_G_l1 + valid_config.lamda * G_mse_loss.item()
                    val_loss_G_ssim = val_loss_G_ssim + valid_config.nu * G_ssim_loss.item()

                    val_loss_D_real = val_loss_D_real + D_real_loss.item()
                    val_loss_D_fake = val_loss_D_fake + D_fake_loss.item()
                # save test
                if flag <= 1:
                    utils.save_image(
                        val_target,
                        "%s/inputAF_epoch_%03d_layer%02d.png" % (im_save_path, epoch, dp + 1)
                    )
                    flag += 1

                utils.save_image(
                    val_fake,
                    "%s/network_epoch_%03d_layer%02d.png" % (im_save_path, epoch, dp + 1)
                )

                val_loss_G_list.append(val_loss_G / counter)
                val_loss_G_l1_list.append(val_loss_G_l1 / counter)
                val_loss_G_ssim_list.append(val_loss_G_ssim / counter)
                val_loss_D_real_list.append(val_loss_D_real / counter)
                val_loss_D_fake_list.append(val_loss_D_fake / counter)
                valid_loss_D_list.append((val_loss_D_real + val_loss_D_fake) * 0.5)

                val_loss_G = 0
                val_loss_G_l1 = 0
                val_loss_G_ssim = 0
                val_loss_D_real = 0
                val_loss_D_fake = 0
                counter = 0
        text = (
                "[%d/%d] 1:Loss_G: %.4f Loss_G_l1: %.4f Loss_G_ssim: %.4f | Loss_D: %.4f D(x): %.4f D(G(z)): %.4f \n"
                "        2:Loss_G: %.4f Loss_G_l1: %.4f Loss_G_ssim: %.4f | Loss_D: %.4f D(x): %.4f D(G(z)): %.4f \n"
                % (
                    epoch,
                    niter,
                    run_loss_G_list[0],
                    run_loss_G_l1_list[0],
                    run_loss_G_ssim_list[0],
                    run_loss_D_list[0],
                    run_loss_D_real_list[0],
                    run_loss_D_fake_list[0],
                    run_loss_G_list[1],
                    run_loss_G_l1_list[1],
                    run_loss_G_ssim_list[1],
                    run_loss_D_list[1],
                    run_loss_D_real_list[1],
                    run_loss_D_fake_list[1],
                )
        )
        print(text)

        valid_text = (
                "[%d/%d] 1:Valid_Loss_G: %.4f Valid_Loss_G_l1: %.4f Valid_Loss_G_ssim: %.4f | Valid_Loss_D: %.4f Valid_D(x): %.4f Valid_D(G(z)): %.4f\n"
                "        2:Valid_Loss_G: %.4f Valid_Loss_G_l1: %.4f Valid_Loss_G_ssim: %.4f | Valid_Loss_D: %.4f Valid_D(x): %.4f Valid_D(G(z)): %.4f\n"
                " Time: %d s"
                % (
                    epoch,
                    niter,
                    val_loss_G_list[0],
                    val_loss_G_l1_list[0],
                    val_loss_G_ssim_list[0],
                    valid_loss_D_list[0],
                    val_loss_D_real_list[0],
                    val_loss_D_fake_list[0],
                    val_loss_G_list[1],
                    val_loss_G_l1_list[1],
                    val_loss_G_ssim_list[1],
                    valid_loss_D_list[1],
                    val_loss_D_real_list[1],
                    val_loss_D_fake_list[1],
                    int(time() - start_time),
                )
        )
        print(valid_text)
        # save model
        torch.save(
            netG.state_dict(), "%s/netG_epoch_%d.pth" % (model_save_path, epoch)
        )
        torch.save(
            netD.state_dict(), "%s/netD_epoch_%d.pth" % (model_save_path, epoch)
        )
