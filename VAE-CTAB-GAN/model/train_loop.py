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
import matplotlib as plt

from model.pipeline.inverse_transform import apply_activate
from model.synthesizer.transformer import ImageTransformer, DataTransformer
from model.condvec import Condvec
from model.sampler import Sampler
from model.model import Classifier, VAEEncoder, Generator

def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)[0]
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

def compute_info_loss(real_features, fake_features, delta_mean=0.1, delta_var=0.1):
    """
    Computes feature-wise information loss.
    - real_features, fake_features: Discriminator 중간 출력 (flattened)
    """
    real_mean = torch.mean(real_features, dim=0)
    fake_mean = torch.mean(fake_features, dim=0)
    real_var = torch.var(real_features, dim=0)
    fake_var = torch.var(fake_features, dim=0)

    mean_loss = torch.sum(F.relu(torch.abs(real_mean - fake_mean) - delta_mean))
    var_loss = torch.sum(F.relu(torch.abs(real_var - fake_var) - delta_var))
    return mean_loss + var_loss

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

def has_nan(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"🚨 NaN detected in {name}!")
    if torch.isinf(tensor).any():
        print(f"⚠️  Inf detected in {name}!")

def train_vae_gan(encoder, generator, discriminator, full_data, cont_data, args, device):
    with open(args.transformer_path, 'rb') as f:
        transformer = pickle.load(f)
    #transformer.output_dim = full_data.shape[1]

    cond_generator = Condvec(full_data, transformer.output_info)
    sampler = Sampler(full_data, transformer.output_info)

    image_size = int(np.ceil(np.sqrt(full_data.shape[1] + cond_generator.n_opt)))
    G_transformer = ImageTransformer(image_size, orig_dim=args.output_dim)
    D_transformer = ImageTransformer(image_size)

    dside = image_size  # ImageTransformer에서 쓰이는 사이즈
    num_channels = 64
    num_classes = cond_generator.n_opt

    classifier = Classifier(dside=dside, num_channels=num_channels, num_classes=num_classes).to(device)
    optimizerC = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    dataset = TensorDataset(torch.tensor(full_data, dtype=torch.float32),
                            torch.tensor(cont_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizerE = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    optimizerG = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        inner_bar = tqdm(enumerate(loader), total=len(loader), desc="Step", leave=False)

        if epoch == args.encoder_freeze_epoch:
            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()

        for step, (x_full, x_cont) in inner_bar:
            x_full = x_full.to(device)
            x_cont = x_cont.to(device)

            c, m, col, opt = cond_generator.sample_train(args.batch_size)
            #print(f"[DEBUG] Conditional vector shape: {c.shape}, z vector shape: {x_cont.shape[0]}")
            if not isinstance(c, torch.Tensor):
                c = torch.from_numpy(c)
            c = c.to(device)

            # 디버깅 코드 - VAE sampling이 안정적인지 확인
            """print(f"[DEBUG] mu.mean={mu.mean().item():.4f}, mu.std={mu.std().item():.4f}")
            print(f"[DEBUG] logvar.mean={logvar.mean().item():.4f}, logvar.std={logvar.std().item():.4f}")
            print(f"[DEBUG] z.mean={z.mean().item():.4f}, z.std={z.std().item():.4f}")"""

            z, mu, logvar = encoder(x_cont)
            input_gen = torch.cat([z, c], dim=1)
            fake_image = generator(input_gen)

            fake_tabular = G_transformer.inverse_transform(fake_image)
            fake_activated = apply_activate(fake_tabular, transformer.output_info)
            recon_cont = fake_tabular[:, :x_cont.shape[1]]
            recon_loss = F.mse_loss(recon_cont, x_cont)
            kl_loss = kl_divergence(mu, logvar)

            if epoch < args.encoder_freeze_epoch:
                vae_loss = recon_loss + args.kl_weight * kl_loss
                optimizerE.zero_grad()
                optimizerG.zero_grad()
                vae_loss.backward()
                optimizerE.step()
                optimizerG.step()

                wandb.log({
                    "epoch": epoch,
                    "step": step,
                    "kl_loss": kl_loss.item(),
                    "recon_loss": recon_loss.item(),
                })
                continue  # skip GAN part
            
            #CTAB-GAN+ 논문 학습 5:1 비율 혹은 10:1 비율
            if step % 5 == 0:
                with torch.no_grad():
                    z, mu, logvar = encoder(x_cont)
                    input_gen = torch.cat([z, c], dim=1)
                    fake_image = generator(input_gen).detach()
                    fake_tabular = G_transformer.inverse_transform(fake_image)
                    fake_activated = apply_activate(fake_tabular, transformer.output_info)
                    fake_cat = torch.cat([fake_activated, c], dim=1)

                    real_data = torch.from_numpy(sampler.sample(args.batch_size, col, opt)).to(device)
                    if epoch < args.real_activate_until_epoch:
                        real_data = apply_activate(real_data, transformer.output_info)
                    real_cat = torch.cat([real_data, c], dim=1)

                    real_image = D_transformer.transform(real_cat)
                    fake_image_d = D_transformer.transform(fake_cat)

                real_validity, _ = discriminator(real_image)
                fake_validity, _ = discriminator(fake_image_d)
                gp = compute_gradient_penalty(discriminator, real_image.data, fake_image_d.data, device)

                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gp
                optimizerD.zero_grad()
                d_loss.backward()
                optimizerD.step()
            else:
                d_loss = torch.tensor(0.0)  # placeholder for wandb logging

            z, mu, logvar = encoder(x_cont)
            input_gen = torch.cat([z, c], dim=1)
            fake_image = generator(input_gen)
            fake_tabular = G_transformer.inverse_transform(fake_image)
            fake_activated = apply_activate(fake_tabular, transformer.output_info)
            fake_cat = torch.cat([fake_activated, c], dim=1)
            fake_image_d = D_transformer.transform(fake_cat)

            ##g_loss
            g_out, _ = discriminator(fake_image_d)
            g_loss = -torch.mean(g_out)
            conditional_loss = cond_loss(fake_tabular, transformer.output_info, c, m)

            ## info_loss
            _, real_features = discriminator(real_image) 
            _, fake_features = discriminator(fake_image_d)

            info_loss = compute_info_loss(real_features, fake_features,
                                        delta_mean=args.delta_mean,
                                        delta_var=args.delta_var)
            
            ##cls loss
            if isinstance(fake_image_d, tuple):  
                fake_image_d = fake_image_d[0]
            advcls_loss = F.cross_entropy(classifier(fake_image_d), torch.argmax(c, dim=1))

            recon_weight = 0.01
            kl_weight = 1.0
            advcls_weight = 1.0
            
            ## freeze 단계에서 kl, recon 영향력 줄이기.
            total_loss = (args.g_weight * g_loss +
                          kl_weight * kl_loss +
                          recon_weight * recon_loss +
                          args.cond_weight * conditional_loss +
                          args.info_weight * info_loss + 
                          advcls_weight * advcls_loss
                          )

            optimizerG.zero_grad()
            optimizerC.zero_grad()
            total_loss.backward()
            optimizerG.step()
            optimizerC.step()

            """# 학습 루프 내부에 삽입
            if step % 100 == 0 and epoch == 0:
                with torch.no_grad():
                    print(f"[DEBUG] G_output: {fake_image[0, :10]}")
                    print(f"[DEBUG] fake_tabular: {fake_tabular[0, :10]}")
                    print(f"[DEBUG] fake_activated: {fake_activated[0, :10]}")"""

            #if step % 100 == 0:
            wandb.log({
                    "epoch": epoch,
                    "step": step,
                    "d_loss": d_loss.item(),
                    "g_loss": g_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "recon_loss": recon_loss.item(),
                    "advcls_loss": advcls_loss.item(),
                    "info_loss": info_loss.item(),
                    "cond_loss": conditional_loss.item(),
                    "total_loss": total_loss.item()
            })
                #print(f"[Epoch {epoch}/{args.epochs}] [Step {step}] D: {d_loss.item():.4f}, G: {g_loss.item():.4f}, KL: {kl_loss.item():.4f}, Recon: {recon_loss.item():.4f}, ADVCLS: {advcls_loss.item():.4f}")

        """min_val = fake_activated.min().item()
        max_val = fake_activated.max().item()
        mean_val = fake_activated.mean().item()
        print(f"[DEBUG] fake_activated range: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")

        print(f"[DEBUG] Shapes -> G_output: {fake_image.shape}, fake_tabular: {fake_tabular.shape}, fake_activated: {fake_activated.shape}")"""

        if (epoch + 1)>=50 and (epoch + 1) % 20 == 0:
            os.makedirs("./checkpoints", exist_ok=True)
            save_path = f"./checkpoints/vae_ctabgan_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'encoder_state_dict': encoder.state_dict()
            }, save_path)
            print(f"Saved checkpoint (G + E) at epoch {epoch + 1} -> {save_path}")

        final_dir = os.path.join(args.checkpoint_dir, "final")
        os.makedirs(final_dir, exist_ok=True)     

    torch.save({
        'encoder': encoder.state_dict(),
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'classifier': classifier.state_dict()
    }, os.path.join(final_dir, f"vae_ctabgan_epoch{epoch+1}.pth"))
    print(f" Final checkpoint saved to {os.path.join(final_dir, f've_ctabgan_epoch{epoch}.pth')}")

def generate_samples(args, full_data, cont_data, device):
    import pickle

    # Load transformer
    with open(args.transformer_path, 'rb') as f:
        transformer = pickle.load(f)
    output_info = transformer.output_info

    # Load DataPrep for inverse_prep
    with open("preprocess/dataprep/dataprep.pkl", "rb") as f:
        dataprep = pickle.load(f)

    condvec = Condvec(full_data, output_info)
    image_size = int(np.ceil(np.sqrt(full_data.shape[1] + condvec.n_opt)))
    transformer_G = ImageTransformer(image_size, orig_dim=args.output_dim)

    encoder = VAEEncoder(input_dim=cont_data.shape[1], latent_dim=args.latent_dim).to(device)
    generator = Generator(input_dim=args.latent_dim + condvec.n_opt,
                          gside=image_size, num_channels=64).to(device)

    checkpoint_path = os.path.join(args.checkpoint_dir, args.save_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    #encoder.load_state_dict(checkpoint['encoder_state_dict']) ## 1epoch할 때는 encoder_state_dict
    #generator.load_state_dict(checkpoint['generator_state_dict'])
    encoder.load_state_dict(checkpoint['encoder']) ## 1epoch할 때는 encoder_state_dict
    generator.load_state_dict(checkpoint['generator'])
    encoder.eval()
    generator.eval()

    samples = []
    for _ in tqdm(range((args.num_samples + args.batch_size - 1) // args.batch_size), desc="Generating"):
        c, _, _, _ = condvec.sample_train(args.batch_size)
        if not isinstance(c, torch.Tensor):
            c = torch.from_numpy(c)
        c = c.to(device)
        z = torch.randn((args.batch_size, args.latent_dim)).to(device)
        input_gen = torch.cat([z, c], dim=1)
        with torch.no_grad():
            fake_image = generator(input_gen) #이미지 데이터로 만들고
            fake_tabular = transformer_G.inverse_transform(fake_image) #이미지를 -> Tabular하게 만들고
            #fake_activated = apply_activate(fake_tabular, output_info) #실수화되어있는 값들을, 원본 데이터와 비슷하게 복원
            samples.append(fake_tabular.cpu().numpy())

    final_samples = np.concatenate(samples, axis=0)[:args.num_samples]
    tabular_data = transformer.inverse_transform(final_samples)  # One-hot → numeric/categorical 복원
    tabular_data = np.where(tabular_data < 0, 0.0, tabular_data)
    recovered_df = dataprep.inverse_prep(tabular_data)  # log, label decoding, rounding 등 최종 복원

    output_path = os.path.join(args.sample_dir, "generated_samples_smotified.csv")
    recovered_df.to_csv(output_path, index=False)
    print(f"✅ Generated {args.num_samples} samples and saved to {output_path}")

