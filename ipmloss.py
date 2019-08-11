import torch
import torch.nn as nn

class IPMLoss(nn.Module):
    def __init__(self,
            ragan, lsgan, importance_sampling,
            torch_dtype, competitive=False
            ):
        super().__init__()

        self.loss = nn.MSELoss() if lsgan else nn.L1Loss()

        self.ragan = ragan
        self.importance_sampling = importance_sampling
        if not self.ragan:
            return

        self.register_buffer('valid', torch.tensor(1., dtype=torch_dtype))
        self.register_buffer('fake', torch.tensor(0. if competitive else -1., dtype=torch_dtype))

    def __call__(self, fake, real, label, n_labels):
#        out = """
        last = 0
        loss = []
        for _ in range(n_labels):
            # replace this with itter trough fake_rollout.label_cuts()
            for w in range(last+1, len(label)):
                if any(label[w-1] != label[w]):
                    break
            idx = list(range(last,w))
            last = w
            l_a = nn.CosineEmbeddingLoss()(fake[idx], real[idx].mean(0, keepdim=True), -torch.ones(fake[idx].shape[0]).to(fake.device))
            l_b = nn.CosineEmbeddingLoss()(real[idx], fake[idx].mean(0, keepdim=True), -torch.ones(real[idx].shape[0]).to(fake.device))

            loss.append(l_a + l_b)
        return sum(loss) / len(loss)
        return 10 * (l_a + l_b) / 2.# + torch.log(
#                1e-8 + torch.exp(-fake.mean(1)).sum() + torch.exp(-real.mean(1)).sum())
#        """
        ed_fake = (fake - real.mean(0, keepdim=True))
        ed_fake = ed_fake.mean(1, keepdim=True)
        ed_real = (real - fake.mean(0, keepdim=True))
        ed_real = ed_real.mean(1, keepdim=True)

        z = 0. if not self.importance_sampling else torch.log(
                1e-8 + torch.exp(-fake).sum() + torch.exp(-real).sum())

        if self.ragan:
            return z + ( # ragan + lsgan
                    self.loss(ed_fake, self.fake.expand_as(ed_fake)) +
                    self.loss(ed_real, self.valid.expand_as(ed_real))
                    ) / 2
        # wgan + importance sampling -> softmax gan
        return z + ed_real.mean()
