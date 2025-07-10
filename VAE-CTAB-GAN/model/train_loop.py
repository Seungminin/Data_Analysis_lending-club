import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, BatchNorm2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss)
from torch.utils.data import DataLoader, TensorDataset
import wandb
import os
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd

from model.pipeline.inverse_transform import apply_activate
from model.synthesizer.transformer import ImageTransformer, DataTransformer
from model.condvec import Condvec
from model.sampler import Sampler
from model.model import CTABClassifier, VAEEncoder, CTABGenerator

def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones_like(d_interpolates).to(device)
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



def cond_loss(data, output_info, c, m):
    
    """
    Used to compute the conditional loss for ensuring the generator produces the desired category as specified by the conditional vector

    Inputs:
    1) data -> raw data synthesized by the generator 
    2) output_info -> column informtion corresponding to the data transformer
    3) c -> conditional vectors used to synthesize a batch of data
    4) m -> a matrix to identify chosen one-hot-encodings across the batch

    Outputs:
    1) loss -> conditional loss corresponding to the generated batch 

    """
    
    # used to store cross entropy loss between conditional vector and all generated one-hot-encodings
    tmp_loss = []
    # counter to iterate generated data columns
    st = 0
    # counter to iterate conditional vector
    st_c = 0
    # iterating through column information
    for item in output_info:
        # ignoring numeric columns
        if item[1] == 'tanh':
            st += item[0]
            continue
        # computing cross entropy loss between generated one-hot-encoding and corresponding encoding of conditional vector
        elif item[1] == 'softmax':
            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
            data[:, st:ed],
            torch.argmax(c[:, st_c:ed_c], dim=1),
            reduction='none')
            tmp_loss.append(tmp)
            st = ed
            st_c = ed_c

    # computing the loss across the batch only and only for the relevant one-hot-encodings by applying the mask 
    tmp_loss = torch.stack(tmp_loss, dim=1)
    loss = (tmp_loss * m).sum() / data.size()[0]

    return loss

def weights_init(model):
    
    """
    This function initializes the learnable parameters of the convolutional and batch norm layers

    Inputs:
    1) model->  network for which the parameters need to be initialized
    
    Outputs:
    1) network with corresponding weights initialized using the normal distribution
    
    """
    
    classname = model.__class__.__name__
    
    if classname.find('Conv') != -1:
        init.normal_(model.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(model.weight.data, 1.0, 0.02)
        init.constant_(model.bias.data, 0)


def get_st_ed(target_col_index,output_info):
    
    """
    Used to obtain the start and ending positions of the target column as per the transformed data to be used by the classifier 

    Inputs:
    1) target_col_index -> column index of the target column used for machine learning tasks (binary/multi-classification) in the raw data 
    2) output_info -> column information corresponding to the data after applying the data transformer

    Outputs:
    1) starting (st) and ending (ed) positions of the target column as per the transformed data
    
    """
    # counter to iterate through columns
    st = 0
    # counter to check if the target column index has been reached
    c= 0
    # counter to iterate through column information
    tc= 0
    # iterating until target index has reached to obtain starting position of the one-hot-encoding used to represent target column in transformed data
    for item in output_info:
        # exiting loop if target index has reached
        if c==target_col_index:
            break
        if item[1]=='tanh':
            st += item[0]
        elif item[1] == 'softmax':
            st += item[0]
            c+=1 
        tc+=1    
    
    # obtaining the ending position by using the dimension size of the one-hot-encoding used to represent the target column
    ed= st+output_info[tc][0] 
    
    return (st,ed)

def train_vae_gan(encoder, generator, discriminator, full_data, cont_data, args, device):
    with open(args.transformer_path, 'rb') as f:
        transformer = pickle.load(f)
    transformer.output_dim = full_data.shape[1]

    cond_generator = Condvec(full_data, transformer.output_info)
    sampler = Sampler(full_data, transformer.output_info)

    image_size = int(np.ceil(np.sqrt(full_data.shape[1] + cond_generator.n_opt)))
    G_transformer = ImageTransformer(image_size)
    D_transformer = ImageTransformer(image_size)

    classifier = CTABClassifier().to(device)
    optimizerC = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    dataset = TensorDataset(torch.tensor(full_data, dtype=torch.float32),
                            torch.tensor(cont_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizerE = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    optimizerG = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        for step, (x_full, x_cont) in enumerate(loader):
            x_full = x_full.to(device)
            x_cont = x_cont.to(device)

            c, m, col, opt = cond_generator.sample_train(args.batch_size)
            c = torch.from_numpy(c).to(device)

            z, mu, logvar = encoder(x_cont)
            input_gen = torch.cat([z, c], dim=1)

            recon_image = generator(input_gen)
            recon_tabular = G_transformer.inverse_transform(recon_image)
            recon_cont = recon_tabular[:, :x_cont.shape[1]]
            recon_loss = F.mse_loss(recon_cont, x_cont)

            fake_image = generator(input_gen).detach()
            fake_tabular = G_transformer.inverse_transform(fake_image)
            fake_activated = apply_activate(fake_tabular, transformer.output_info)

            real_data = torch.from_numpy(sampler.sample(args.batch_size, col, opt)).to(device)
            real_cat = torch.cat([real_data, c], dim=1)
            fake_cat = torch.cat([fake_activated, c], dim=1)

            real_image = D_transformer.transform(real_cat)
            fake_image_d = D_transformer.transform(fake_cat)

            real_validity = discriminator(real_image)
            fake_validity = discriminator(fake_image_d)
            gp = compute_gradient_penalty(discriminator, real_image.data, fake_image_d.data, device)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gp
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            z, mu, logvar = encoder(x_cont)
            input_gen = torch.cat([z, c], dim=1)
            fake_image = generator(input_gen)
            fake_tabular = G_transformer.inverse_transform(fake_image)
            fake_activated = apply_activate(fake_tabular, transformer.output_info)
            fake_cat = torch.cat([fake_activated, c], dim=1)
            fake_image_d = D_transformer.transform(fake_cat)

            g_loss = -torch.mean(discriminator(fake_image_d))
            kl_loss = kl_divergence(mu, logvar)
            advcls_loss = F.cross_entropy(classifier(fake_image_d), torch.argmax(c, dim=1))

            total_loss = (args.g_weight * g_loss +
                          args.kl_weight * kl_loss +
                          args.recon_weight * recon_loss +
                          advcls_loss)

            optimizerG.zero_grad()
            optimizerE.zero_grad()
            optimizerC.zero_grad()
            total_loss.backward()
            optimizerG.step()
            optimizerE.step()
            optimizerC.step()

            if step % 100 == 0:
                wandb.log({
                    "epoch": epoch,
                    "step": step,
                    "d_loss": d_loss.item(),
                    "g_loss": g_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "recon_loss": recon_loss.item(),
                    "advcls_loss": advcls_loss.item(),
                    "total_loss": total_loss.item()
                })
                print(f"[Epoch {epoch}/{args.epochs}] [Step {step}] D: {d_loss.item():.4f}, G: {g_loss.item():.4f}, KL: {kl_loss.item():.4f}, Recon: {recon_loss.item():.4f}, ADVCLS: {advcls_loss.item():.4f}")

        torch.save({
            'encoder': encoder.state_dict(),
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'classifier': classifier.state_dict()
        }, os.path.join(args.checkpoint_dir, f"vae_ctabgan_epoch{epoch}.pth"))

def generate_samples(args, full_data, cont_data, device):
    with open(args.transformer_path, 'rb') as f:
        transformer = pickle.load(f)
    output_info = transformer.output_info

    condvec = Condvec(full_data, output_info)
    image_size = int(np.ceil(np.sqrt(full_data.shape[1] + condvec.n_opt)))
    transformer_G = ImageTransformer(image_size)

    encoder = VAEEncoder(input_dim=cont_data.shape[1], latent_dim=args.latent_dim).to(device)
    generator = CTABGenerator(latent_dim=args.latent_dim + condvec.n_opt).to(device)

    checkpoint_path = os.path.join(args.checkpoint_dir, args.save_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    generator.load_state_dict(checkpoint['generator'])
    encoder.eval()
    generator.eval()

    samples = []
    for _ in range((args.num_samples + args.batch_size - 1) // args.batch_size):
        c, _, _, _ = condvec.sample_train(args.batch_size)
        c = torch.from_numpy(c).to(device)
        z = torch.randn((args.batch_size, args.latent_dim)).to(device)
        input_gen = torch.cat([z, c], dim=1)
        with torch.no_grad():
            fake_image = generator(input_gen)
            fake_tabular = transformer_G.inverse_transform(fake_image)
            fake_activated = apply_activate(fake_tabular, output_info)
            samples.append(fake_activated.cpu().numpy())

    final_samples = np.concatenate(samples, axis=0)[:args.num_samples]
    output_path = os.path.join(args.sample_dir, "generated_samples.csv")
    pd.DataFrame(final_samples).to_csv(output_path, index=False)
    print(f"✅ Generated {args.num_samples} samples and saved to {output_path}")
