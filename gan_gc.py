import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class GeneratorCritic(nn.Module):
    def __init__(self, generator, critic):
        super().__init__()
        self.generator = generator
        self.critic = critic

        for i, a in enumerate(self.generator):
            self.add_module("generator_%i"%i, a)
        for i, c in enumerate(self.critic):
            self.add_module("critic_%i"%i, c)

#use itertools.chain instead...
#      itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    def parameters(self):
        return np.concatenate([
                self.generator_parameters(),
                np.concatenate([
                    list(critic.parameters()) for critic in self.critic], 0)],
                0)

    def generator_parameters(self):
        return np.concatenate([list(generator.parameters()) for generator in self.generator], 0)

    def critic_parameters(self, ind):
        c_i = ind if ind < len(self.critic) else 0
        return self.critic[c_i].parameters()

    def forward(self, noise, space, label, ind = 0):
        assert ind != -1 or len(self.critic) == 1, "you forgot to specify which critic should be used"
        total = len(noise)
        noise = noise.view(-1, len(self.generator), noise.shape[-1])
        space = space.view(-1, len(self.generator), space.shape[-1])
        label = label.view(-1, len(self.generator), label.shape[-1])

        fakes = []
        for i in range(noise.size(1)):
            a_i = (i % len(self.generator)) if noise.size(1) > 1 or 1 == len(self.generator) else random.randint(0, len(self.generator)-1)
            out = self.generator[a_i](noise[:, i, :], space[:, i, :], label[:, i, :])
            fakes.append(out)
        fakes = torch.cat(fakes, 1)

        fakes = fakes.view(total, -1)
        return fakes, self.critic[ind](fakes)

    def eval(self, fakes, ind):
        return self.critic[ind](fakes)

    def gen(self, noise, space, label, ind):
        noise = noise.view(-1, noise.shape[-1])
        space = space.view(-1, space.shape[-1])
        label = label.view(-1, label.shape[-1])
        ind = ind % len(self.generator)
        return self.generator[ind](noise, space, label)
