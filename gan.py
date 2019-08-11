import random

import numpy as np

import torch
from torch.autograd import Variable
import torch.autograd as autograd

from gan_gc import GeneratorCritic


# DRAGAN
def compute_gradient_penalty(gc, x, latent_dim, c):
    alpha = torch.tensor(np.random.random((x.size(0), 1, 1, 1)),
            dtype=x.dtype).to(x.device)

    interpolates = alpha * x + (
            (1. - alpha) * (x + .5 * x.std() * torch.rand(x.size()).to(alpha.device)))
    interpolates = Variable(interpolates, requires_grad=True)

    d_interpolates, _, _ = gc.eval(interpolates, c)
    fake = torch.ones(x.shape[0], latent_dim).to(alpha.device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = (gradients.norm(2, dim=1) - 1)
    gradient_penalty = (gradient_penalty ** 2)
    return gradient_penalty.mean()

class GAN:
    def __init__(self,
            device,
            gan_loss,
        # models
            GenFactory, n_generators, g_callback,
            Critic, n_critics, lipshits_sync,
        # backprop params
            lr_generator, lr_critic, lr_info,
            beta1, beta2,
            clip_norm,
        # training config
            noise_consistent,
            g_batch_size, c_batch_size,
            n_steps, c_train_n_step,
        # GAN config
            lambda_gp,
            lambda_space, lambda_label, lambda_gt,
            latent_dim, n_labels
            ):

        self.device = device
        self.latent_dim = latent_dim
        self.n_labels = n_labels

        self.t = 0
        self.c = [0] * n_critics
        self.lipshits_sync = lipshits_sync

        self.noise_consistent = noise_consistent

        self.g_batch_size = g_batch_size
        self.c_batch_size = c_batch_size

        self.n_steps = n_steps
        self.c_train_n_step = c_train_n_step

        self.clip_norm = clip_norm

        self.lambda_gp = lambda_gp
        self.gan_loss = gan_loss

        self.feature_matching_v2_loss = torch.nn.CosineEmbeddingLoss()#SmoothL1Loss()#MSELoss()
        self.space_loss = torch.nn.L1Loss()
        self.label_loss = torch.nn.CrossEntropyLoss()

        self.lambda_space = lambda_space
        self.lambda_label = lambda_label
        self.lambda_gt = lambda_gt

        self.g_callback = g_callback

        gf = GenFactory()
        self.gc = GeneratorCritic(
                    [ torch.nn.DataParallel(gf.head().to(device)) for _ in range(n_generators) ],
#                    [ gf.head().to(device) for _ in range(n_generators) ],
#                    [ torch.nn.DataParallel(Critic().to(device)) for _ in range(n_critics) ])
                    [ Critic().to(device) for _ in range(n_critics) ])

        self.optimizer_G = torch.optim.Adam(self.gc.generator_parameters(), lr=lr_generator, betas=(beta1, beta2))
        self.optimizer_C = [
                torch.optim.Adam(critic.parameters(), lr=lr_critic, betas=(beta1, beta2)
                    ) for critic in self.gc.critic ]
        self.optimizer_I = torch.optim.Adam(self.gc.parameters(), lr=lr_info, betas=(beta1, beta2))

    def learn(self, fake_rollout, data, labels=None):
        for _ in range(self.n_steps):
            self._step(fake_rollout, data, labels)

        return self.c_loss, self.g_loss, self.i_loss, self.l_loss

    def _step(self, fake_rollout, data, labels):
        self.t += 1
        # prepare data to train on
        fake_noise, fake_space, fake_label = fake_rollout.batch(self.g_batch_size)
        real_data = data[
                fake_rollout.indices if self.noise_consistent else fake_rollout.get_indices(
                    len(data), self.g_batch_size)
                ].squeeze(0)
        label_cuts = fake_rollout.label_cuts()

        for gen in self.gc.generator:
            gen.train()
        # EVALUATE GENERATOR + UPDATE CRITIC
        g_loss = self._train(real_data, labels, fake_noise, fake_space, fake_label, label_cuts)
        self.g_loss = g_loss.detach().cpu().numpy()

        for c in range(len(self.gc.critic)):
            # UPDATE INTERNAL REPRESENTATION of data!!
            self._internal_representation_loss(c, data, fake_rollout, labels)

            # ENFORCE 1-LIPSHITZ-nes
            self._enforce_1lipshitz(c, data, self.latent_dim)

    def _train(self, real_data, label, fake_noise, fake_space, fake_label, label_cuts):
        g_loss = []
        for c in range(len(self.gc.critic)):
            # eval GENERTOR
            loss, fake_data, real_vals = self._eval_generator(c, real_data, fake_noise, fake_space, fake_label, label_cuts)
            g_loss.append(
                    self.g_callback(
                        self, c,
                        real_data, label,
                        fake_data, fake_noise, fake_space, fake_label,
                        loss))
            # eval + update CRITIC
            self._train_critic(c, real_data, fake_data, real_vals, fake_label, label_cuts)
            # UPDATE GENERATOR -> as multiple heads we want to do it after critic
            self._backpropagate(self.optimizer_G, self.gc.generator_parameters(), g_loss[-1]) # learn!!
        return sum(g_loss) / len(g_loss)

    def _train_critic(self, c, real_data, fake_data, real_vals, fake_label, label_cuts):
        self.c_loss = torch.zeros(1)
        # CRITIC
        for _ in range(self.c_train_n_step):
            c_loss = self._eval_critic(c, fake_data.detach(), real_vals, fake_label, label_cuts)
            loss = c_loss # loss = self.c_callback(real_data, fake_data, c_loss)
            self._backpropagate(self.optimizer_C[c], self.gc.critic_parameters(c), loss, True) # learn!!
            self.c_loss = loss.detach().cpu().numpy()

            # ENFORCE 1-LIPSHITZ-nes
            self.c[c] += 1
            if not self.lipshits_sync or self.c[c] % self.lipshits_sync:
                continue
            self._enforce_1lipshitz(c, real_data, self.latent_dim)

    def _eval_critic(self, c, fake_data, real_validity, fake_label, label_cuts):
        ''' IPM method for critic learning by default '''
        indices = list(range(len(fake_data))) if self.c_batch_size == self.g_batch_size else random.sample(
                range(len(fake_data)), self.c_batch_size)

        fake_validity, _, _ = self.gc.eval(fake_data[indices], c)

        return self.gan_loss.backward_loss(label_cuts, fake_validity, real_validity[indices])

    def _eval_generator(self, c, real_data, fake_noise, fake_space, fake_label, label_cuts):
        ''' we using here matching PatchGAN features, real vs fake, as function of critic with multiple values '''

        real_vals, _, _ = self.gc.eval(real_data, c)
        fake_data, (fake_vals, _, _) = self.gc(fake_noise, fake_space, fake_label)
        loss = self.gan_loss.forward_loss(label_cuts, fake_vals, real_vals)

        return loss, fake_data, real_vals

    def _internal_representation_loss(self, c, data, fake_rollout, labels):
        ''' labeling internally for finer generation ( semi-supervised learning ) '''
        # info loss - on full rollout batch
        fakes, (_, pred_space, pred_label) = self.gc(
                fake_rollout.noise, fake_rollout.space, fake_rollout.one_hot_label)

        space_loss = self.lambda_space * self.space_loss(pred_space, fake_rollout.space)
        label_loss = self.lambda_label * self.label_loss(pred_label, fake_rollout.label)
        loss = label_loss + space_loss

        if labels is not None: # if we want labels corespond our perception
            _, _, pred_label = self.gc.eval(data, c)
            #  _, _, pred_label = self.gc.eval(data[fake_rollout.indices], 0)
#            label = torch.tensor(labels, dtype=torch.long).to(pred_label.device)
            label = labels.to(pred_label.device)
            gt_loss = self.lambda_gt * self.label_loss(pred_label, label)
            loss += gt_loss

        self._backpropagate(self.optimizer_I, self.gc.parameters(), loss)#, True)
        self.i_loss = loss.detach().cpu().numpy()

    def _enforce_1lipshitz(self, c, data, latent_dim):
        gp = compute_gradient_penalty(self.gc, data.data, latent_dim, c)
        loss = (gp * self.lambda_gp)
        self._backpropagate(self.optimizer_C[c], self.gc.critic_parameters(c), loss, True)
        self.l_loss = gp.detach().cpu().numpy()

    def _backpropagate(self, optimizer, params, loss, retain_graph=False):
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        torch.nn.utils.clip_grad_norm_(params, self.clip_norm)
        optimizer.step()
