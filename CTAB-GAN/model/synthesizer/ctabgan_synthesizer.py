import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, BatchNorm2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss)
from model.synthesizer.transformer import ImageTransformer,DataTransformer
from tqdm import tqdm
import wandb

import os 


def random_choice_prob_index_sampling(probs,col_idx):
    
    """
    Used to sample a specific category within a chosen one-hot-encoding representation 
                                                        
    Inputs:
    1) probs -> probability mass distribution of categories 
    2) col_idx -> index used to identify any given one-hot-encoding
    
    Outputs:
    1) option_list -> list of chosen categories 
    
    """

    option_list = []
    for i in col_idx:
        # for improved stability
        pp = probs[i] + 1e-6 
        pp = pp / sum(pp)
        # sampled based on given probability mass distribution of categories within the given one-hot-encoding 
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    
    return np.array(option_list).reshape(col_idx.shape)

class Condvec:
    def __init__(self, data, output_info, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = []
        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        self.p_log_sampling = []
        self.p_sampling = []

        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                self.interval.append((self.n_opt, item[0]))
                self.n_col += 1
                self.n_opt += item[0]
                freq = np.sum(data[:, st:ed], axis=0)
                log_freq = np.log(freq + 1)
                self.p_log_sampling.append(torch.tensor(log_freq / np.sum(log_freq), dtype=torch.float32, device=self.device))
                self.p_sampling.append(torch.tensor(freq / np.sum(freq), dtype=torch.float32, device=self.device))
                st = ed

        self.interval = torch.tensor(self.interval, device=self.device)

    def sample_train(self, batch):
        if self.n_col == 0:
            return None

        vec = torch.zeros((batch, self.n_opt), dtype=torch.float32, device=self.device)
        mask = torch.zeros((batch, self.n_col), dtype=torch.float32, device=self.device)
        idx = torch.randint(0, self.n_col, (batch,), device=self.device)
        mask[torch.arange(batch, device=self.device), idx] = 1

        opt1prime = torch.empty(batch, dtype=torch.long, device=self.device)
        for i in range(batch):
            p = self.p_log_sampling[idx[i]] + 1e-6
            p = p / torch.sum(p)
            opt1prime[i] = torch.multinomial(p, 1).item()

        for i in range(batch):
            start = self.interval[idx[i], 0]
            vec[i, start + opt1prime[i]] = 1

        return vec, mask, idx, opt1prime

    def sample(self, batch, fraud_type=None):
        if self.n_col == 0:
            return None

        vec = torch.zeros((batch, self.n_opt), dtype=torch.float32, device=self.device)

        if fraud_type is not None:
            idx = torch.full((batch,), self.get_fraud_type_index(fraud_type), dtype=torch.long, device=self.device)
        else:
            idx = torch.randint(0, self.n_col, (batch,), device=self.device)

        for i in range(batch):
            p = self.p_sampling[idx[i]] + 1e-6
            p = p / torch.sum(p)
            sampled = torch.multinomial(p, 1).item()
            start = self.interval[idx[i], 0]
            vec[i, start + sampled] = 1

        return vec.cpu().numpy()  # still returning numpy for now
    def get_fraud_type_index(self, fraud_type):
        mapping = {
            "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5,
            "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12
        }
        return mapping.get(fraud_type, 0)


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

class Sampler:
    def __init__(self, data, output_info, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.tensor(data, dtype=torch.float32, device=self.device)
        self.n = len(data)
        self.model = []

        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(torch.nonzero(self.data[:, st + j]).squeeze(1))
                self.model.append(tmp)
                st = ed

    def sample(self, n, col, opt):
        if col is None:
            idx = torch.randint(0, self.n, (n,), device=self.device)
            return self.data[idx].cpu().numpy()

        sample_indices = []
        for c, o in zip(col, opt):
            try:
                candidates = self.model[c][o]
                if len(candidates) == 0:
                    raise IndexError("No candidates found.")
                idx = torch.randint(0, len(candidates), (1,), device=self.device).item()
                sample_indices.append(candidates[idx].item())
            except (IndexError, RuntimeError):
                fallback = torch.randint(0, self.n, (1,), device=self.device).item()
                sample_indices.append(fallback)

        return self.data[sample_indices].cpu().numpy()



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

class Classifier(Module):

    """
    This class represents the classifier module used along side the discriminator to train the generator network

    Variables:
    1) dim -> column dimensionality of the transformed input data after removing target column
    2) class_dims -> list of dimensions used for the hidden layers of the classifier network
    3) str_end -> tuple containing the starting and ending positions of the target column in the transformed input data

    Methods:
    1) __init__() -> initializes and builds the layers of the classifier module 
    2) forward() -> executes the forward pass of the classifier module on the corresponding input data and
                    outputs the predictions and corresponding true labels for the target column 
    
    """
    
    def __init__(self,input_dim, class_dims,st_ed):
        super(Classifier,self).__init__()
        # subtracting the target column size from the input dimensionality 
        self.dim = input_dim-(st_ed[1]-st_ed[0])
        # storing the starting and ending positons of the target column in the input data
        self.str_end = st_ed
        
        # building the layers of the network with same hidden layers as discriminator
        seq = []
        tmp_dim = self.dim
        for item in list(class_dims):
            seq += [
                Linear(tmp_dim, item),
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            tmp_dim = item
        
        # in case of binary classification the last layer outputs a single numeric value which is squashed to a probability with sigmoid
        if (st_ed[1]-st_ed[0])==2:
            seq += [Linear(tmp_dim, 1),Sigmoid()]
        # in case of multi-classs classification, the last layer outputs an array of numeric values associated to each class
        else: seq += [Linear(tmp_dim,(st_ed[1]-st_ed[0]))] 
            
        self.seq = Sequential(*seq)

    def forward(self, input):
        
        # true labels obtained from the input data
        label = torch.argmax(input[:, self.str_end[0]:self.str_end[1]], axis=-1)
        
        # input to be fed to the classifier module
        new_imp = torch.cat((input[:,:self.str_end[0]],input[:,self.str_end[1]:]),1)
        
        # returning predictions and true labels for binary/multi-class classification 
        if ((self.str_end[1]-self.str_end[0])==2):
            return self.seq(new_imp).view(-1), label
        else: return self.seq(new_imp), label

class Discriminator(Module):

    """
    This class represents the discriminator network of the model

    Variables:
    1) seq -> layers of the network used for making the final prediction of the discriminator model
    2) seq_info -> layers of the discriminator network used for computing the information loss

    Methods:
    1) __init__() -> initializes and builds the layers of the discriminator model
    2) forward() -> executes a forward pass on the input data to output the final predictions and corresponding 
                    feature information associated with the penultimate layer used to compute the information loss 
    
    """
    
    def __init__(self, layers):
        super(Discriminator, self).__init__()
        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[:len(layers)-2])

    def forward(self, input):
        return (self.seq(input)), self.seq_info(input)

class Generator(Module):
    
    """
    This class represents the discriminator network of the model
    
    Variables:
    1) seq -> layers of the network used by the generator

    Methods:
    1) __init__() -> initializes and builds the layers of the generator model
    2) forward() -> executes a forward pass using noise as input to generate data 

    """
    
    def __init__(self, layers):
        super(Generator, self).__init__()
        self.seq = Sequential(*layers)

    def forward(self, input):
        return self.seq(input)

def determine_layers_disc(side, num_channels):
    
    """
    This function describes the layers of the discriminator network as per DCGAN (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

    Inputs:
    1) side -> height/width of the input fed to the discriminator
    2) num_channels -> no. of channels used to decide the size of respective hidden layers 

    Outputs:
    1) layers_D -> layers of the discriminator network
    
    """

    # computing the dimensionality of hidden layers 
    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        # the number of channels increases by a factor of 2 whereas the height/width decreases by the same factor with each layer
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    # constructing the layers of the discriminator network based on the recommendations mentioned in https://arxiv.org/abs/1511.06434 
    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]
    # last layer reduces the output to a single numeric value which is squashed to a probabability using sigmoid function
    layers_D += [
        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), 
        Sigmoid() 
    ]
    
    return layers_D

def determine_layers_gen(side, random_dim, num_channels):
    
    """
    This function describes the layers of the generator network
    
    Inputs:
    1) random_dim -> height/width of the noise matrix to be fed for generation 
    2) num_channels -> no. of channels used to decide the size of respective hidden layers

    Outputs:
    1) layers_G -> layers of the generator network

    """
    if side is None:
        raise ValueError("side value cannot be None")
    
    # computing the dimensionality of hidden layers
    layer_dims = [(1, side), (num_channels, side // 2)]
    
    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))
    
    # similarly constructing the layers of the generator network based on the recommendations mentioned in https://arxiv.org/abs/1511.06434 
    # first layer of the generator takes the channel dimension of the noise matrix to the desired maximum channel size of the generator's layers 
    layers_G = [
        ConvTranspose2d(
            random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)
    ]
    
    # the following layers are then reversed with respect to the discriminator 
    # such as the no. of channels reduce by a factor of 2 and height/width of generated image increases by the same factor with each layer 
    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)
        ]

    return layers_G

def apply_activate(data, output_info):
    
    """
    This function applies the final activation corresponding to the column information associated with transformer

    Inputs:
    1) data -> input data generated by the model in the same format as the transformed input data
    2) output_info -> column information associated with the transformed input data

    Outputs:
    1) act_data -> resulting data after applying the respective activations 

    """
    
    data_t = []
    # used to iterate through columns
    st = 0
    # used to iterate through column information
    for item in output_info:
        # for numeric columns a final tanh activation is applied
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        # for one-hot-encoded columns, a final gumbel softmax (https://arxiv.org/pdf/1611.01144.pdf) is used 
        # to sample discrete categories while still allowing for back propagation 
        elif item[1] == 'softmax':
            ed = st + item[0]
            # note that as tau approaches 0, a completely discrete one-hot-vector is obtained
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
    
    act_data = torch.cat(data_t, dim=1) 

    return act_data

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

class CTABGANSynthesizer:

    """
    This class represents the main model used for training the model and generating synthetic data


    Variables:
    1) random_dim -> size of the noise vector fed to the generator
    2) class_dim -> tuple containing dimensionality of hidden layers for the classifier network
    3) num_channels -> no. of channels for deciding respective hidden layers of discriminator and generator networks
    4) dside -> height/width of the input data fed to discriminator network
    5) gside -> height/width of the input data generated by the generator network
    6) l2scale -> parameter to decide strength of regularization of the network based on constraining l2 norm of weights
    7) batch_size -> no. of records to be processed in each mini-batch of training
    8) epochs -> no. of epochs to train the model
    9) device -> type of device to be used for training (i.e., gpu/cpu)
    10) generator -> generator network from which data can be generated after training the model

    Methods:
    1) __init__() -> initializes the model with user specified parameters
    2) fit() -> takes the pre-processed training data and associated parameters as input to fit the CTABGANSynthesizer model 
    3) sample() -> takes as input the no. of data rows to be generated and synthesizes the corresponding no. of data rows

    """ 
    
    def __init__(self,
                 class_dim=(128, 128, 128, 128),
                 random_dim=100,
                 num_channels=8,
                 l2scale=1e-5,
                 batch_size=100,
                 device = None,
                 epochs=1):
                 
        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.dside = None
        self.gside = None
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = None
        self.transformer = None  # Ensure transformer is initialized here\
    
    def _build_generator_only(self):
        """
        Used for sample-only mode (e.g., --generate): initialize only the generator 
        using self.output_info and self.cond_generator.
        """
        data_dim = self.transformer.output_dim
        sides = [64, 128, 256, 512]
        col_size_g = data_dim
        for i in sides:
            if i * i >= col_size_g:
                self.gside = i
                break
        else:
            raise ValueError(f"Data dimension too high for given sides: {col_size_g}")
        
        layers_G = determine_layers_gen(self.gside, self.random_dim + self.cond_generator.n_opt, self.num_channels)
        self.generator = Generator(layers_G).to(self.device)
        self.Gtransformer = ImageTransformer(self.gside)

    def fit(self, train_data, transformer, condvec, sampler, output_info, problem_type=None, target_index=None):
        self.transformer = transformer
        self.cond_generator = condvec
        data_sampler = sampler

        data_dim = transformer.output_dim

        # set image sides
        sides = [64, 128, 256, 512]
        col_size_d = data_dim + self.cond_generator.n_opt
        for i in sides:
            if i * i >= col_size_d:
                self.dside = i
                break
        else:
            raise ValueError(f"Data dimension too high for given sides: {col_size_d}")

        col_size_g = data_dim
        for i in sides:
            if i * i >= col_size_g:
                self.gside = i
                break
        else:
            raise ValueError(f"Data dimension too high for given sides: {col_size_g}")

        # models
        layers_G = determine_layers_gen(self.gside, self.random_dim + self.cond_generator.n_opt, self.num_channels)
        layers_D = determine_layers_disc(self.dside, self.num_channels)
        self.generator = Generator(layers_G).to(self.device)
        discriminator = Discriminator(layers_D).to(self.device)

        optimizer_params = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)

        st_ed = None
        classifier = None
        optimizerC = None
        if target_index is not None:
            st_ed = get_st_ed(target_index, output_info)
            classifier = Classifier(data_dim, self.class_dim, st_ed).to(self.device)
            optimizerC = optim.Adam(classifier.parameters(), **optimizer_params)

        self.generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.Gtransformer = ImageTransformer(self.gside)
        self.Dtransformer = ImageTransformer(self.dside)

        steps_per_epoch = max(1, len(train_data) // self.batch_size)

        for i in tqdm(range(self.epochs), desc="Epoch"):
            for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {i}", leave=False):
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                c, m, col, opt = self.cond_generator.sample_train(self.batch_size)
                c = c if isinstance(c, torch.Tensor) else torch.from_numpy(c)
                c = c.to(self.device)
                m = m if isinstance(m, torch.Tensor) else torch.from_numpy(m)
                m = m.to(self.device)
                noisez = torch.cat([noisez, c], dim=1).view(self.batch_size, -1, 1, 1)

                perm = np.arange(self.batch_size)
                np.random.shuffle(perm)
                real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                real = torch.from_numpy(real.astype('float32')).to(self.device)
                c_perm = c[perm]

                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake).to(self.device)
                fakeact = apply_activate(faket, self.transformer.output_info).to(self.device)

                fake_cat = torch.cat([fakeact, c], dim=1)
                real_cat = torch.cat([real, c_perm], dim=1)
                real_cat_d = self.Dtransformer.transform(real_cat)
                fake_cat_d = self.Dtransformer.transform(fake_cat)

                optimizerD.zero_grad()
                y_real, _ = discriminator(real_cat_d)
                y_fake, _ = discriminator(fake_cat_d)
                loss_d = -(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean())
                loss_d.backward()
                optimizerD.step()

                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                c = c if isinstance(c, torch.Tensor) else torch.from_numpy(c)
                c = c.to(self.device)
                m = m if isinstance(m, torch.Tensor) else torch.from_numpy(m)
                m = m.to(self.device)
                noisez = torch.cat([noisez, c], dim=1).view(self.batch_size, -1, 1, 1)

                optimizerG.zero_grad()
                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake).to(self.device)
                fakeact = apply_activate(faket, self.transformer.output_info).to(self.device)
                fake_cat = torch.cat([fakeact, c], dim=1)
                fake_cat = self.Dtransformer.transform(fake_cat)

                y_fake, info_fake = discriminator(fake_cat)
                _, info_real = discriminator(real_cat_d)
                cross_entropy = cond_loss(faket, self.transformer.output_info, c, m)
                g = -(torch.log(y_fake + 1e-4).mean()) + cross_entropy
                g.backward(retain_graph=True)

                loss_mean = torch.norm(torch.mean(info_fake.view(self.batch_size,-1), dim=0) - torch.mean(info_real.view(self.batch_size,-1), dim=0), 1)
                loss_std = torch.norm(torch.std(info_fake.view(self.batch_size,-1), dim=0) - torch.std(info_real.view(self.batch_size,-1), dim=0), 1)
                loss_info = loss_mean + loss_std
                loss_info.backward()
                optimizerG.step()

                if problem_type:
                    c_loss = BCELoss() if (st_ed[1] - st_ed[0]) == 2 else CrossEntropyLoss()

                    optimizerC.zero_grad()
                    real_pre, real_label = classifier(real)
                    if (st_ed[1] - st_ed[0]) == 2:
                        real_label = real_label.type_as(real_pre)
                    loss_cc = c_loss(real_pre, real_label)
                    loss_cc.backward()
                    optimizerC.step()

                    optimizerG.zero_grad()
                    fake = self.generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake).to(self.device)
                    fakeact = apply_activate(faket, self.transformer.output_info).to(self.device)
                    fake_pre, fake_label = classifier(fakeact)
                    if (st_ed[1] - st_ed[0]) == 2:
                        fake_label = fake_label.type_as(fake_pre)
                    loss_cg = c_loss(fake_pre, fake_label)
                    loss_cg.backward()
                    optimizerG.step()

                    if wandb.run is not None:
                        wandb.log({"loss_classifier_real": loss_cc.item(), "loss_classifier_fake": loss_cg.item()})


            if wandb.run is not None:
                wandb.log({"loss_classifier_real": loss_cc.item(), "loss_classifier_fake": loss_cg.item()})

            
            if (i+1) >= 20 and (i+1) % 20 == 0:
                os.makedirs("./checkpoints", exist_ok=True)
                save_path = f"./checkpoints/ctabgan_epoch_{i+1}.pt"
                torch.save({
                    'epoch': i+1,
                    "generator_state_dict" : self.generator.state_dict(), 
                }, save_path)
                print(f"saved checkpoint at epoch {i+1} -> {save_path}")
                
                            
    def sample(self, num_samples, fraud_types: list):        
        self.generator.eval()
        output_info = self.transformer.output_info
        
        all_data = []
        for fraud_type in fraud_types:
            data = []
            steps = num_samples//self.batch_size + 1
            for _ in tqdm(range(steps), desc ="Generating"): 
                # generating synthetic data using sampled noise and conditional vectors
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample(self.batch_size, fraud_type = fraud_type)
                c = condvec
                c = torch.from_numpy(c).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake).to(self.device)
                fakeact = apply_activate(faket,output_info)
                data.append(fakeact.detach().cpu().numpy())

            data = np.concatenate(data, axis=0)

        # applying the inverse transform and returning synthetic data in a similar form as the original pre-processed training data
            result = self.transformer.inverse_transform(data)
            all_data.append(result[0:num_samples])

        final_data = np.concatenate(all_data, axis=0)

        return final_data

