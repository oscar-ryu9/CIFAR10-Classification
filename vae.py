
#Part 1: Variational Autoencoder + Conditional Variational Autoencoder

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
# torch.cuda.is_available = lambda: False
from ssim import SSIM

def compute_score(loss, min_thres, max_thres):
    if loss <= min_thres:
        base_score = 100.0
    elif loss >= max_thres:
        base_score = 0.0
    else:
        base_score = (1 - float(loss - min_thres) / (max_thres - min_thres)) \
                     * 100
    return base_score

# -----
# VAE Build Blocks



class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        in_channels: int = 3,
        ):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        self.conv1 = nn.Conv2d(self.in_channels, 32, 3) #30
        self.conv2 = nn.Conv2d(32, 64, 3)               #28
        self.maxp1 = nn.MaxPool2d(2, 2)                 #14
        self.conv3 = nn.Conv2d(64, 128, 3)              #12
        self.maxp2 = nn.MaxPool2d(2, 2)                 #6
        self.conv4 = nn.Conv2d(128, 256, 3)             #4
        self.maxp3 = nn.MaxPool2d(2, 2)                 #2

        self.fc1 = nn.Linear(256*2*2, self.latent_dim)
        self.fc2 = nn.Linear(256*2*2, self.latent_dim)

    
    def forward(self, x):
        
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.maxp1(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.maxp2(x)
        x = F.leaky_relu(self.conv4(x))
        x = self.maxp3(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc1(x)
        log_var = self.fc2(x)
        
        return mu, log_var
    

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 3,
        ):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        
        self.fc1 = nn.Linear(self.latent_dim, 256*2*2)
        self.trans0 = nn.ConvTranspose2d(256, 256, 9) #8
        self.trans1 = nn.ConvTranspose2d(256, 128, 7) #14
        self.trans2 = nn.ConvTranspose2d(128, 128, 5) #18
        self.trans3 = nn.ConvTranspose2d(128, 64, 5)  #22
        self.trans4 = nn.ConvTranspose2d(64, 64, 5)   #26
        self.trans5 = nn.ConvTranspose2d(64, 32, 3)   #28
        self.trans6 = nn.ConvTranspose2d(32, 32, 3)   #30
        self.trans7 = nn.ConvTranspose2d(32, self.out_channels, 1)
        
    def forward(self, z):

        z = self.fc1(z)
        z = z.view(-1, 256, 2, 2)
        z = F.leaky_relu(self.trans0(z))
        z = F.leaky_relu(self.trans1(z))
        z = F.leaky_relu(self.trans2(z))
        z = F.leaky_relu(self.trans3(z))
        z = F.leaky_relu(self.trans4(z))
        z = F.leaky_relu(self.trans5(z))
        z = F.leaky_relu(self.trans6(z))
        z = F.leaky_relu(self.trans7(z))
        z = torch.sigmoid(z)

        return z

# #####
# Wrapper for Variational Autoencoder
# #####

class VAE(nn.Module):
    def __init__(
        self, 
        latent_dim: int = 128,
        ):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encode = Encoder(latent_dim=latent_dim)
        self.decode = Decoder(latent_dim=latent_dim)

    def reparameterize(self, mu, log_var):
        """Reparameterization Tricks to sample latent vector z
        from distribution w/ mean and variance.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x, y):
        # #####
        # : Complete forward for VAE
        # #####

        #encode
        mu, log_var = self.encode(x)
        #reparam
        z = self.reparameterize(mu, log_var)
        #decode
        xg = self.decode(z)

        """Forward for CVAE.
        Returns:
            xg: reconstructed image from decoder.
            mu, log_var: mean and log(std) of z ~ N(mu, sigma^2)
            z: latent vector, z = mu + sigma * eps, acquired from reparameterization trick. 
        """

        return xg, mu, log_var, z

    def generate(
        self,
        n_samples: int,
        ):

        """Randomly sample from the latent space and return
        the reconstructed samples.
        Returns:
            xg: reconstructed image
            None: a placeholder simply.
        """
        #xg init
        xg = torch.empty(n_samples, 3, 32, 32)

        #loop here
        for i in range(n_samples):

            #latent space + GPU Activation
            fake_space = torch.randn(self.latent_dim)
            if torch.cuda.is_available():
                fake_space = fake_space.cuda()
            
            # new decode
            xg[i] = self.decode(fake_space)

        return xg, None


# #####
# Wrapper for Conditional Variational Autoencoder
# #####

class CVAE(nn.Module):
    def __init__(
        self, 
        latent_dim: int = 128,
        num_classes: int = 10,
        img_size: int = 32,
        ):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size

        self.encode = Encoder(latent_dim=latent_dim, in_channels=3)
        self.decode = Decoder(latent_dim=latent_dim+10)



    def reparameterize(self, mu, log_var):
        """Reparameterization Tricks to sample latent vector z
        from distribution w/ mean and variance.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x, y):
        """Forward for CVAE.
        Returns:
            xg: reconstructed image from decoder.
            mu, log_var: mean and log(std) of z ~ N(mu, sigma^2)
            z: latent vector, z = mu + sigma * eps, acquired from reparameterization trick. 
        """
        #encode
        mu, log_var = self.encode(x)

        #conditional one hot + GPU activation
        one_hot_vector = F.one_hot(y, self.num_classes)
        if torch.cuda.is_available():
            one_hot_vector = one_hot_vector.cuda()
        
        #reparam
        z = self.reparameterize(mu, log_var)

        #concat
        z = torch.cat((z, one_hot_vector), dim=1)

        #decode
        xg = self.decode(z)

        return xg, mu, log_var, z

    def generate(
        self,
        n_samples: int,
        y: torch.Tensor = None,
        ):
        """Randomly sample from the latent space and return
        the reconstructed samples.
        NOTE: Randomly generate some classes here, if not y is provided.
        Returns:
            xg: reconstructed image
            y: classes for xg. 
        """
        if y is None:
            y = torch.randint(0, self.num_classes, (n_samples,))

        # xg init
        xg = torch.empty(n_samples, 3, 32, 32)

        #loop here
        for i in range(n_samples):

            #latent space and conditional one hot
            fake_space = torch.randn(self.latent_dim)
            one_hot_vector = F.one_hot(y[i], self.num_classes)

            # GPU activation
            if torch.cuda.is_available():
                fake_space = fake_space.cuda()
                one_hot_vector = one_hot_vector.cuda()
            
            #concat
            z = torch.cat((fake_space, one_hot_vector), dim=0)

            #new decode
            xg[i] = self.decode(z)
        
        return xg, y


# #####
# Wrapper for KL Divergence
# #####

class KLDivLoss(nn.Module):
    def __init__(
        self,
        lambd: float = 1.0,
        ):
        super(KLDivLoss, self).__init__()
        self.lambd = lambd

    def forward(
        self, 
        mu, 
        log_var,
        ):
        loss = 0.5 * torch.sum(-log_var - 1 + mu ** 2 + log_var.exp(), dim=1)
        self.lambd = min(0.001, self.lambd)
        return self.lambd * torch.mean(loss)

def main():
    # -----
    # Hyperparameters
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # NOTE: Feel free to change the hyperparameters as long as you meet the marking requirement
    # NOTE: DO NOT TRAIN IT LONGER THAN 100 EPOCHS.
    batch_size = 256
    workers = 2
    latent_dim = 128
    lr = 0.0005
    num_epochs = 60
    validate_every = 1
    print_every = 100

    conditional = False     # Flag to use VAE or CVAE

    if conditional:
        name = "cvae"
    else:
        name = "vae"

    # Set up save paths
    if not os.path.exists(os.path.join(os.path.curdir, "visualize", name)):
        os.makedirs(os.path.join(os.path.curdir, "visualize", name))
    save_path = os.path.join(os.path.curdir, "visualize", name)
    ckpt_path = name + '.pt'

    kl_annealing = [0, 0.0001] + [(i+1)/(num_epochs-2) for i in range(num_epochs-2)]     # KL Annealing


    # -----
    # Dataset
    # NOTE: Data is only normalized to [0, 1]. THIS IS IMPORTANT!!!
    tfms = transforms.Compose([
        transforms.ToTensor(),
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,
        transform=tfms)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True,
        transform=tfms,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=workers)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=workers)

    subset = torch.utils.data.Subset(
        test_dataset, 
        [0, 380, 500, 728, 1000, 2300, 3400, 4300, 4800, 5000])

    loader = torch.utils.data.DataLoader(
        subset, 
        batch_size=10)

    # -----
    # Model
    if conditional:
        model = CVAE(latent_dim=latent_dim)
    else:
        model = VAE(latent_dim=latent_dim)

    
    l2_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    ssim_criterion = SSIM()
    KLDiv_criterion = KLDivLoss()



    best_total_loss = float("inf")

    # Send to GPU
    if torch.cuda.is_available():
        model = model.cuda()



    optimizer = optim.Adam(model.parameters(), lr=lr)

    # To further help with training
    # NOTE: You can remove this if you find this unhelpful
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [40, 50], gamma=0.1, verbose=False)


    # -----
    # Train loop


    def train_step(x, y):
        """One train step for VAE/CVAE.
        You should return average total train loss(sum of reconstruction losses, kl divergence loss)
        and all individual average reconstruction loss (l2, bce, ssim) per sample.
        Args:
            x, y: one batch (images, labels) from Cifar10 train set.
        Returns:
            loss: total loss per batch.
            l2_loss: MSE loss for reconstruction.
            bce_loss: binary cross-entropy loss for reconstruction.
            ssim_loss: ssim loss for reconstruction.
            kldiv_loss: kl divergence loss.
        """
        xg, mu, log_var, _ = model(x, y)
        x = x.cpu()
        xg = xg.cpu()

        optimizer.zero_grad()
        l2_loss = l2_criterion(xg, x)
        bce_loss = bce_criterion(xg, x)
        ssim_loss = 1-ssim_criterion(xg, x)
        kldiv_loss = KLDiv_criterion(mu, log_var)
        loss = l2_loss + bce_loss + ssim_loss + kldiv_loss
        loss = loss.cpu()
        loss.backward()
        optimizer.step()

        return loss, l2_loss, bce_loss, ssim_loss, kldiv_loss



    def denormalize(x):
        """Denomalize a normalized image back to uint8.
        Args:
            x: torch.Tensor, in [0, 1].
        Return:
            x_denormalized: denormalized image as numpy.uint8, in [0, 255].
        """
        x *= 255
        x = x.transpose(1, 3).transpose(1, 2).cpu().detach().numpy()
        x_denormalized = x.astype(np.uint8)
        return x_denormalized

    # Loop HERE
    l2_losses = []
    bce_losses = []
    ssim_losses = []
    kld_losses = []
    total_losses_test = []

    total_losses_train = []

    for epoch in range(1, num_epochs + 1):
        total_loss_train = 0.0
        for i, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # Train step
            model.train()
            loss, recon_loss, bce_loss, ssim_loss, kldiv_loss = train_step(x, y)
            total_loss_train += loss * x.shape[0]
            
            # Print
            if i % print_every == 0:
                print("Epoch {}, Iter {}: Total Loss: {:.6f} MSE: {:.6f}, SSIM: {:.6f}, BCE: {:.6f}, KLDiv: {:.6f}".format(epoch, i, loss, recon_loss, ssim_loss, bce_loss, kldiv_loss))

        total_losses_train.append(total_loss_train / len(train_dataset))

        # Test loop
        if epoch % validate_every == 0:
            # Loop through test set
            model.eval()


            with torch.no_grad():
                for x, y in test_loader:
                    if torch.cuda.is_available():
                        x = x.cuda()
                        y = y.cuda()

                    xg, mu, log_var, _ = model(x, y)

                    xg = xg.cpu()
                    x = x.cpu()
                    l2_loss = l2_criterion(x, xg)
                    bce_loss = bce_criterion(xg, x)
                    ssim_loss = (1 - ssim_criterion(x, xg)).cpu()
                    kld_loss = KLDiv_criterion(mu, log_var).cpu()
                    l2_losses.append(l2_loss)
                    bce_losses.append(bce_loss)
                    ssim_losses.append(ssim_loss)
                    kld_losses.append(kld_loss)
                    
                    avg_total_recon_loss_test = l2_loss + bce_loss + ssim_loss
                    total_losses_test.append(avg_total_recon_loss_test + kld_loss)

                # Plot losses
                if epoch > 1:
                    plt.plot(l2_losses, label="L2 Reconstruction")
                    plt.plot(bce_losses, label="BCE")
                    plt.plot(ssim_losses, label="SSIM")
                    plt.plot(kld_losses, label="KL Divergence")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.xlim([1, epoch])
                    plt.legend()
                    plt.savefig(os.path.join(os.path.join(save_path, "losses.png")), dpi=300)
                    plt.clf()
                    plt.close('all')

                    plt.plot(total_losses_test, label="Total Loss Test")
                    plt.plot(total_losses_train, label="Total Loss Train")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.xlim([1, epoch])
                    plt.legend()
                    plt.savefig(os.path.join(os.path.join(save_path, "total_loss.png")), dpi=300)
                    plt.clf()
                    plt.close('all')
                
                # Save best model
                if avg_total_recon_loss_test < best_total_loss and epoch > 50:
                    torch.save(model.state_dict(), ckpt_path)
                    best_total_loss = avg_total_recon_loss_test
                    print("Best model saved w/ Total Reconstruction Loss of {:.6f}.".format(best_total_loss))

            # Do some reconstruction
            model.eval()
            with torch.no_grad():
                x, y = next(iter(loader))
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                #y_onehot = F.one_hot(y, 10).float()
                xg, _, _, _ = model(x, y)

                # Visualize
                xg = denormalize(xg)
                x = denormalize(x)

                y = y.cpu().numpy()

                plt.figure(figsize=(10, 5))
                for p in range(10):
                    plt.subplot(4, 5, p+1)
                    plt.imshow(xg[p])
                    plt.subplot(4, 5, p + 1 + 10)
                    plt.imshow(x[p])
                    plt.text(0, 0, "{}".format(classes[y[p].item()]), color='black',
                                backgroundcolor='white', fontsize=8)
                    plt.axis('off')

                plt.savefig(os.path.join(os.path.join(save_path, "E{:d}.png".format(epoch))), dpi=300)
                plt.clf()
                plt.close('all')
                print("Figure saved at epoch {}.".format(epoch))

        # KL Annealing
        # Adjust scalar for KL Divergence loss
        KLDiv_criterion.lambd = kl_annealing[epoch-1]

        print("Lambda:", KLDiv_criterion.lambd)
        
        # LR decay
        scheduler.step()
        
        print()

    # Generate some random samples
    if conditional:
        model = CVAE(latent_dim=latent_dim)
    else:
        model = VAE(latent_dim=latent_dim)
    if torch.cuda.is_available():
        model = model.cuda()
    ckpt = torch.load(name+'.pt')
    model.load_state_dict(ckpt)

    # Generate 20 random images
    xg, y = model.generate(20)
    xg = denormalize(xg)
    if y is not None:
        y = y.cpu().numpy()

    plt.figure(figsize=(10, 5))
    for p in range(20):
        plt.subplot(4, 5, p+1)
        if y is not None:
            plt.text(0, 0, "{}".format(classes[y[p].item()]), color='black',
                    backgroundcolor='white', fontsize=8)
        plt.imshow(xg[p])
        plt.axis('off')

    plt.savefig(os.path.join(os.path.join(save_path, "random.png")), dpi=300)
    plt.clf()
    plt.close('all')

    if conditional:
        min_val, max_val = 0.92, 1.0
    else:
        min_val, max_val = 0.92, 1.0

    print("Total reconstruction loss:", best_total_loss)
    print("Min Val:", min_val)
    print("Max Val:", max_val)


if __name__ == '__main__':
    main()