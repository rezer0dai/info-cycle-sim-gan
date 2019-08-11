from __future__ import print_function
import argparse
import os
import numpy as np
import math
import sys
import random
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

from models import *
from gan import GAN
from ipmloss import *
from rollout import FakeRollout
from utils import *

class GeneratorFactory:
    def __init__(self, layers):
        #  self.factory = NoisyNetFactory(layers, DEVICE)
        pass
    def head(self):
        return GeneratorResNet(
                [opt.channels, opt.img_size, opt.img_size],
#                4, opt.space_dim+opt.n_classes)#,
                5, opt.space_dim+opt.n_classes)#,
                #  self.factory.head(),
                #  1000, 1, opt.space_dim+opt.n_classes)


class CycleRollout(FakeRollout):
    def __init__(self, data, space_dim, Tensor):
        self.noise = data
        self.space = Tensor(np.random.uniform(-1, 1, (len(self.noise), space_dim)))
        self.label = torch.cat([
            torch.ones(len(data) // 2, dtype=torch.long),
            torch.zeros(len(data) // 2, dtype=torch.long)
            ]).to(self.space.device)

        self._one_hot_label(2)

    def _sample_indices(self, batch_size):
        return random.sample(
                range(len(self.noise)//2-1), batch_size-self.random_cut) + random.sample(
                        range(len(self.noise)//2, len(self.noise)-1), self.random_cut)

    def get_indices(self, size, batch_size):
        return self._sample_indices(batch_size)

    def _get_indices(self, size, batch_size):
        self.random_cut = random.randint(batch_size // 4, batch_size - batch_size // 4 - 1)
        return self._sample_indices(batch_size)

    def label_cuts(self):
        self.cuts = []
        labels = self.label[self.indices]
        last = 0
        for i in range(2):
            for w in range(last+1, len(labels)):
                if labels[w-1] != labels[w]:
                    break
            self.cuts.append( (last, w) )
            idx = list(range(*self.cuts[-1]))
            last = w
        return self.cuts

def main(opt, device, Tensor):
    img_shape = (opt.channels, opt.img_size, opt.img_size)

    transforms_ = [
            transforms.Resize(int(opt.img_size * 1.12), Image.BICUBIC),
            transforms.RandomCrop((opt.img_size, opt.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    dataloader = DataLoader(
            ImageDataset("%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=1,
        )

    transforms_ = [
            transforms.Resize(int(opt.img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    sampler = DataLoader(
        ImageDataset("%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
        batch_size=opt.n_samples,
        shuffle=True,
        num_workers=1,
    )


    print("DATALOADER LEN : ", len(dataloader), opt.batch_size, (opt.batch_size-1) * len(dataloader))

    GAME_NATURE = -1.#0.#
    LAYERS = [opt.latent_dim + opt.space_dim + opt.n_classes, 512, 1024, int(np.prod(img_shape))]
    LAYERS = [
                #  opt.latent_dim + opt.space_dim + opt.n_classes + int(np.prod([56, 56])),
                #  opt.latent_dim + opt.space_dim + opt.n_classes + int(np.prod(img_shape)),
                int(np.prod(img_shape)),
                #  int(np.prod(img_shape)),
                int(np.prod(img_shape))
            ]

    pixel_loss = torch.nn.L1Loss()
    def eta_cycle_callback():
        eta_cycle_callback.checkpoint = time.time()
        def cycle_callback(gan, c, real_data, label, fake_data, fake_noise, fake_space, fake_label, gan_loss):
            b_data = real_data.detach()
            b_label = fake_label.detach()
            a_data = fake_noise.detach().view(b_data.shape)
            a_label = torch.ones(b_label.shape).to(device) - b_label
            ab_data = fake_data.view(b_data.shape)

            #id_a = gan.gc.gen(a_data, fake_space, a_label, 0).view(b_data.shape)
            id_b = gan.gc.gen(b_data, fake_space, b_label, 0).view(b_data.shape)
            #loss_identity = ( pixel_loss(id_a, a_data) + pixel_loss(id_b, b_data) )# / 2.
            loss_identity = pixel_loss(id_b, b_data)

            cycle_data = gan.gc.gen(ab_data, fake_space, a_label, 0).view(real_data.shape)
            loss_cycle = pixel_loss(cycle_data, a_data)

            loss = gan_loss + loss_identity * .5 + loss_cycle * 3.

            todo = opt.n_epochs * (len(dataloader)-1) * gan.n_steps - gan.t
            eta = datetime.timedelta(seconds=todo * (time.time() - eta_cycle_callback.checkpoint))
            eta_cycle_callback.checkpoint = time.time()
            print("\r[{:6d}] CYCLE GAN callback G[{:2f}] ID[{:2f}] C[{:2f}] TOTAL[{:2f}] || ETA : {}".format(
                gan.t, gan_loss.item(), loss_identity.item(), loss_cycle.item(), loss.item(), eta),
                end="")

            return loss
        return cycle_callback

    gan_loss = GANLoss(
            forward=SimForwardLoss().to(device),
            backward=SimBackwardLoss().to(device)
#            backward=IPMBackwardLoss(lsgan=True, importance_sampling=False, competitive=False, torch_dtype=torch.float).to(device)
            )

    gan = GAN(device,#"cuda:0",#
            gan_loss,
            # models
            lambda: GeneratorFactory(LAYERS), 1, eta_cycle_callback(),
            lambda: Discriminator(
                [ opt.channels, opt.img_size, opt.img_size ],
                opt.latent_dim,
                opt.n_classes,
                opt.space_dim,
                n_strided=4), 1, lipshits_sync=0,
            # backprop params
            #  lr_generator=5e-4, lr_critic=7e-5, lr_info=2e-4,
            lr_generator=2e-4, lr_critic=2e-4, lr_info=2e-4,
            beta1=.5, beta2=.999,
            clip_norm=2.,
            # training config
            noise_consistent=False,
            g_batch_size=8, c_batch_size=8,
#            n_steps=10, c_train_n_step=1,#0,#
            n_steps=7, c_train_n_step=1,#0,#
            # GAN config
            lambda_gp=1.,
            lambda_space=.1, lambda_label=.6, lambda_gt=.4,
#            lambda_space=.3, lambda_label=1., lambda_gt=.2,
            latent_dim=opt.latent_dim, n_labels=opt.n_classes
            )

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, batch in enumerate(dataloader):
            if len(batch["A"]) < opt.batch_size:
                continue

            imgs_a = Tensor(batch["A"])
            imgs_b = Tensor(batch["B"])

            noise = torch.cat([ imgs_a, imgs_b ]).detach()
            fake_rollout = CycleRollout(
                    data=noise,
                    space_dim=opt.space_dim,
                    Tensor=Tensor)

            real = torch.cat([ imgs_b, imgs_a ]).detach()
            c_loss, g_loss, i_loss, l_loss = gan.learn(
                fake_rollout, data=real,
                labels=fake_rollout.label.detach())

            # -----------
            # Statistics
            # -----------

            print(
                "\n[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [L loss: %f] [D loss: %f] [I loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), g_loss, l_loss, c_loss, i_loss)
            )


            imgs = next(iter(sampler))
            imgs_a = Tensor(imgs["A"])
            imgs_b = Tensor(imgs["B"])

            # Sample noise, labels and code as generator input
            # Sample noise, labels and code as generator input
            N_COLUMNS = opt.n_samples
            N_ROWS = 3
            to_torch = lambda data: Tensor(data)
            repre = lambda data: np.vstack([data]*N_ROWS)
            adapt = lambda data: to_torch(repre(data)).to(device)

            label_data = repre( np.zeros(N_COLUMNS, dtype=np.int) ).flatten()
            space_data = to_torch(np.vstack(
                np.vstack([ # one space
                    np.random.permutation( np.random.normal(
                        np.linspace(-np.random.normal(p, 2e-1), np.random.normal(p, 2e-1), opt.space_dim), 3e-1 ) )
                    ] * N_COLUMNS) for p in np.linspace(-1, 1, N_ROWS))
                ).view(len(label_data), -1)

            with torch.no_grad():
                fakes_a = gan.gc.gen(imgs_a.repeat(N_ROWS, 1, 1, 1), space_data, Tensor(one_hot(np.ones(label_data.shape, dtype=np.int), opt.n_classes)), 0)
                fakes_b = gan.gc.gen(imgs_b.repeat(N_ROWS, 1, 1, 1), space_data, Tensor(one_hot(np.zeros(label_data.shape, dtype=np.int), opt.n_classes)), 0)

            out_shape = [len(space_data), opt.channels, opt.img_size, opt.img_size]
            img_x_shape = [N_COLUMNS, opt.channels, opt.img_size, opt.img_size]
            fakes = torch.cat([
                imgs_a.reshape(img_x_shape),
                fakes_a.reshape(out_shape),
                imgs_b.reshape(img_x_shape),
                fakes_b.reshape(out_shape)])
            save_image(fakes.data, "images_%s_%s/%d.png" % (opt.dataset_name, opt.postfix, batches_done // opt.sample_interval), nrow=N_COLUMNS, normalize=True)

            with torch.no_grad():
                fake_validity, space, _ = gan.gc.eval(torch.stack([fakes_a[0], fakes_b[0]]), 0)
                real_validity = gan.gc.eval(torch.stack([imgs_a[0], imgs_b[0]]).reshape(2, -1), 0)[0].mean(0)

#            print(fake_validity, fake_validity.mean())
#            print(real_validity, real_validity.mean())
            print(fake_validity.mean(0).mean(), real_validity.mean(), space.mean(0))

            batches_done += 1

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="apple2orange", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--lrg", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--lrc", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--lri", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    parser.add_argument("--latent_dim", type=int, default=64, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
    parser.add_argument("--space_dim", type=int, default=4, help="latent code")

    #  parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")

    parser.add_argument("--sample_interval", type=int, default=10, help="interval betwen image samples")

    parser.add_argument("--postfix", type=str, default="", help="latent code")
    parser.add_argument("--n_samples", type=int, default=12, help="samples to output for visuals")

    opt = parser.parse_args()
    print(opt)

    os.makedirs("images_%s_%s"%(opt.dataset_name, opt.postfix), exist_ok=True)

    device = "cuda"#"cpu"
    Tensor = to_tensor(torch.float, device)
    main(opt, device, Tensor)
