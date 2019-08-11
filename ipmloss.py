import torch
import torch.nn as nn

class GANLoss:
    def __init__(self, forward, backward):
        self.forward = forward
        self.backward = backward

    def forward_loss(self, label_cuts, fake_vals, real_vals):
        return self.forward(label_cuts, fake_vals, real_vals)

    def backward_loss(self, label_cuts, fake_vals, real_vals):
        return self.backward(label_cuts, fake_vals, real_vals)

class SimForwardLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CosineEmbeddingLoss()

    def forward(self, label_cuts, fake_vals, real_vals):
        loss = []
        for a, b in label_cuts:
            idx = list(range(a,b))

            l_a = self.loss(
                        fake_vals[idx].mean(0, keepdim=True),
                        real_vals[idx].detach(), 
                        torch.ones(len(idx)).to(fake_vals.device)
                        ).to(fake_vals.device)
            loss.append(l_a)

            l_b = self.loss(
                        real_vals[idx].detach().mean(0, keepdim=True), 
                        fake_vals[idx],
                        torch.ones(len(idx)).to(fake_vals.device)
                        ).to(fake_vals.device)
            loss.append(l_b)

        loss = [sum(loss) / len(loss)]
        l_ab = self.loss(
                    fake_vals.mean(0, keepdim=True),
                    real_vals.detach().mean(0, keepdim=True), 
                    torch.ones(1).to(fake_vals.device)
                    ).to(fake_vals.device)
        loss.append(l_ab)

        return sum(loss) / len(loss)

class SimBackwardLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CosineEmbeddingLoss()

    def forward(self, label_cuts, fake, real):
        loss = []
        for a, b in label_cuts:
            idx = list(range(a,b))

            l_a = self.loss(fake[idx], real[idx].mean(0, keepdim=True), -torch.ones(fake[idx].shape[0]).to(fake.device))
            l_b = self.loss(real[idx], fake[idx].mean(0, keepdim=True), -torch.ones(real[idx].shape[0]).to(fake.device))

            loss.append(l_a + l_b)
        return sum(loss) / len(loss)

class IPMBackwardLoss(nn.Module):
    def __init__(self,
            lsgan, importance_sampling,
            torch_dtype, competitive=False
            ): # importance_sampling from SoftMax paper, just experiment, need to properly test etc...
        super().__init__()

        self.loss = nn.MSELoss()

        self.ragan = ragan
        self.importance_sampling = importance_sampling
        if not self.lsgan:
            return

        self.register_buffer('valid', torch.tensor(1., dtype=torch_dtype))
        self.register_buffer('fake', torch.tensor(0. if competitive else -1., dtype=torch_dtype))

    def forward(self, _, fake, real):
        ed_fake = (fake - real.mean(0, keepdim=True))
        ed_fake = ed_fake.mean(1, keepdim=True)
        ed_real = (real - fake.mean(0, keepdim=True))
        ed_real = ed_real.mean(1, keepdim=True)

        z = 0. if not self.importance_sampling else torch.log(
                1e-8 + torch.exp(-fake).sum() + torch.exp(-real).sum())

        if self.lsgan:
            return z + ( # ragan + lsgan
                    self.loss(ed_fake, self.fake.expand_as(ed_fake)) +
                    self.loss(ed_real, self.valid.expand_as(ed_real))
                    ) / 2
        # wgan + importance sampling -> softmax gan
        return z + ed_real.mean()
