import argparse
import os
import numpy as np
import math
import sys
# from c_budget_allocation import cal_MI_of_two_dataset, cal_JSD_of_two_dataset
import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from torchvision import datasets
from torch.autograd import Variable
# from generator_discriminator_Trans import Generator,Generator1, OriTrans_Discriminator,LSTM_Generator,LSTMDiscriminator
from DPLTGAN_Framwork import Generator, OriTrans_Discriminator
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import torch
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from utils.rdp_accountant import compute_rdp, get_privacy_spent
from tqdm import tqdm

def load_data(data_path,seq_lenth):
    X_train = np.load(data_path, allow_pickle=True)
    print('Training dataset size', X_train.shape)

    X_train = X_train[:3000, :seq_length, 1:]
    timestamp_column = X_train[:, :, 0]
    timestamp_objects = [list(map(lambda ts: datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S"), ts_list)) for ts_list in
                         timestamp_column]

    years_list = []
    months_list = []
    days_list = []
    hours_list = []
    minutes_list = []
    seconds_list = []

    # Loop through Timestamps and extract components
    for ts_list in timestamp_objects:
        years_list.append([ts.year for ts in ts_list])
        months_list.append([ts.month for ts in ts_list])
        days_list.append([ts.day for ts in ts_list])
        hours_list.append([ts.hour for ts in ts_list])
        minutes_list.append([ts.minute for ts in ts_list])
        seconds_list.append([ts.second for ts in ts_list])

    # Convert lists to NumPy arrays
    years_array = np.array(years_list)
    months_array = np.array(months_list)
    days_array = np.array(days_list)
    hours_array = np.array(hours_list)
    minutes_array = np.array(minutes_list)
    seconds_array = np.array(seconds_list)

    # Stack extracted components with original data
    new_data = np.dstack((X_train, years_array, months_array, days_array, hours_array, minutes_array, seconds_array))
    # Retain original time columns separately
    time_column = new_data[:, :, 0]
    # Delete the time column in the original dataset
    new_data = np.delete(new_data, 0, axis=2)
    new_data = new_data.astype(float)

    print('The time columns processing is complete.')
    return new_data,time_column
def custom_gradient(grad):
    device = grad.device
    fake_grad_norm = torch.norm(grad)
    # clip grad
    clip_coeff = torch.abs(grad).mean()
    if fake_grad_norm > clip_coeff:
        grad.mul_(clip_coeff / fake_grad_norm)

    # add noise
    grad.add_(torch.FloatTensor(grad.size()).normal_(0, sigma * clip_coeff).to(device))
    return grad
if __name__== "__main__":

    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    private=True
    sigma = 2
    data_feature_size = 2
    noise_size = 2
    batch_size = 30
    seq_length=100
    lstm_feature_size = 100
    lstm_hidden_size = 100
    generator = Generator(generated_dim=data_feature_size, noise_dim=2, seq_length=seq_length,
                         embed_dim=60).to(device)
    discriminator = OriTrans_Discriminator(in_channels=8, emb_size=60, seq_length=seq_length,depth=3, n_classes=1).to(device)
    generator.to(device)
    discriminator.to(device)
    if cuda:
        generator.cuda()
        discriminator.cuda()

    # load_data
    data_path = "../T-drive traj_data.npy"
    trajs, time_column = load_data(data_path, seq_length)
    real_trajs = trajs[:,:,:2]
    privacy_trajs = trajs
    trajs_num = trajs.shape[0]
    ss = StandardScaler()
    ss.fit(np.vstack(real_trajs))
    real_trajs = real_trajs - ss.mean_
    real_trajs = torch.cat((torch.FloatTensor(real_trajs), torch.FloatTensor(trajs[:,:,2:])),dim=2)
    dataloader = DataLoader(
        TensorDataset(torch.FloatTensor(real_trajs), torch.FloatTensor(privacy_trajs)),
        batch_size=batch_size, shuffle=True)

    Hyperparams = namedtuple('Hyperparams',
                             ['batch_size', 'micro_batch_size', 'lr', 'clamp_upper', 'clamp_lower', 'clip_coeff', 'sigma',
                              'num_epochs'])

    hyperparams = Hyperparams(batch_size=batch_size, micro_batch_size=1, lr=0.0001,
                              clamp_upper=1000, clamp_lower=-1000, clip_coeff=0.0008, sigma=0.001,
                              num_epochs=5)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    adversarial_loss = torch.nn.BCELoss().to(device)
    traj_loss = torch.nn.MSELoss().to(device)
    one = torch.ones(batch_size, 1).to(device)
    mone = one * 0

    for epoch in range(hyperparams.num_epochs):
        batches_done = 0
        for i, (real_traj, privacy_traj) in enumerate(dataloader):
            real_traj = real_traj.to(device)
            privacy_traj = real_traj.to(device)
            z = torch.cat((torch.randn(batch_size, privacy_traj.shape[1], noise_size).to(device),privacy_traj[:,:,2:]),
                dim=2).to(device)
            # Generate a batch of images
            fake_traj = generator(z)
            optimizer_D.zero_grad()

            # concat time columns for fake_traj
            fake_traj = torch.cat((fake_traj, privacy_traj[:,:,2:]), dim=2).to(device)
            real_loss = adversarial_loss(discriminator(real_traj), one)
            fake_loss = adversarial_loss(discriminator(fake_traj), mone)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # stop the update of discriminator
            for p in discriminator.parameters():
                p.requires_grad = False
            optimizer_G.zero_grad()

            #train Generator
            z = torch.cat((
                          torch.randn(batch_size, privacy_traj.shape[1], noise_size).to(device),
                          privacy_traj[:, :, 2:]),
                          dim=2).to(device)
            fake = generator(z)
            # fake.retain_grad()
            fake_traj = torch.cat((fake, privacy_traj[:, :, 2:]), dim=2).to(device)
            g_loss = adversarial_loss(discriminator(fake_traj), one) + traj_loss(real_traj[:,:,:2]*100, fake_traj[:,:,:2]*100)
            if private:
                # Customizing the backpropagation path
                hook = fake_traj.register_hook(custom_gradient)
                g_loss.backward()
                hook.remove()
            else:
                g_loss.backward()
            optimizer_G.step()


            if private:
                # Calculate the current privacy cost using the accountant
                max_lmbd = 64
                lmbds = range(2, max_lmbd + 1)
                rdp = compute_rdp(batch_size / trajs_num, sigma, epoch*(trajs_num/batch_size)+batches_done, lmbds)
                # print("i",i)
                # print('steps',epoch*(trajs_num/batch_size)+batches_done)
                epsilon, _, _ = get_privacy_spent(lmbds, rdp, target_delta=1e-5)
            else:
                if epoch > hyperparams.num_epochs:
                    epsilon = np.inf
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [epsilon: %f]"
                % (epoch, hyperparams.num_epochs, batches_done % len(dataloader), len(dataloader), d_loss.item(),
                   g_loss.item(),epsilon)
            )

            batches_done += 1

        if epoch%10==0 and epoch !=0:
            # Save models at regular intervals
            torch.save(generator,f"generator-epoch-{epoch}.pth")
            torch.save(discriminator,f"discriminator-epoch-{epoch}.pth")