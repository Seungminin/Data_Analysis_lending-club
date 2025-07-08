
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb
import os


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand(real_samples.size())
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_vae_gan(encoder, generator, discriminator, full_data, cont_data, args, device):
    dataset = TensorDataset(torch.tensor(full_data, dtype=torch.float32),
                            torch.tensor(cont_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizerE = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    optimizerG = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    os.makedirs("./checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        for step, (x_full, x_cont) in enumerate(loader):
            x_full = x_full.to(device)
            x_cont = x_cont.to(device)

            # === Train Discriminator ===
            z, mu, logvar = encoder(x_cont)
            fake_data = generator(torch.cat([z], dim=1)).detach()
            real_validity = discriminator(x_full)
            fake_validity = discriminator(fake_data)
            gp = compute_gradient_penalty(discriminator, x_full.data, fake_data.data, device)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gp

            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            # === Train Generator & VAE ===
            z, mu, logvar = encoder(x_cont)
            fake_data = generator(torch.cat([z], dim=1))
            g_loss = -torch.mean(discriminator(fake_data))
            kl_loss = kl_divergence(mu, logvar)
            total_loss = args.g_weight * g_loss + args.kl_weight * kl_loss

            optimizerG.zero_grad()
            optimizerE.zero_grad()
            total_loss.backward()
            optimizerG.step()
            optimizerE.step()

            if step % 100 == 0:
                wandb.log({
                    "epoch": epoch,
                    "step": step,
                    "d_loss": d_loss.item(),
                    "g_loss": g_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "total_loss": total_loss.item()
                })
                print(f"[Epoch {epoch}/{args.epochs}] [Step {step}] D: {d_loss.item():.4f}, G: {g_loss.item():.4f}, KL: {kl_loss.item():.4f}")

        # Save checkpoint every epoch
        torch.save({
            'encoder': encoder.state_dict(),
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict()
        }, f"./checkpoints/vae_ctabgan_epoch{epoch}.pth")
