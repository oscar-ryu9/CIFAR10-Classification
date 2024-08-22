#Part 2: AC-GAN

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

def compute_score(acc, min_thres, max_thres):
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
                     * 100
    return base_score


# -----
# AC-GAN Build Blocks

class Generator(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.out_channels = out_channels

        self.fc1 = nn.Linear(self.latent_dim+10, 256*2*2)
        self.trans0 = nn.ConvTranspose2d(256, 256, 9) #8
        self.trans1 = nn.ConvTranspose2d(256, 128, 7) #14
        self.trans2 = nn.ConvTranspose2d(128, 128, 5) #18
        self.trans3 = nn.ConvTranspose2d(128, 64, 5)  #22
        self.trans4 = nn.ConvTranspose2d(64, 64, 5)   #26
        self.trans5 = nn.ConvTranspose2d(64, 32, 3)   #28
        self.trans6 = nn.ConvTranspose2d(32, 32, 3)   #30
        self.trans7 = nn.ConvTranspose2d(32, self.out_channels, 1)


        
    def forward(self, z, y):
        one_hot_vector = F.one_hot(y, self.num_classes)
        z = torch.cat((z, one_hot_vector), dim=1)
        z = self.fc1(z)
        z = z.view(-1, 256, 2, 2)
        z = F.leaky_relu(self.trans0(z), 0.2)
        z = F.leaky_relu(self.trans1(z), 0.2)
        z = F.leaky_relu(self.trans2(z), 0.2)
        z = F.leaky_relu(self.trans3(z), 0.2)
        z = F.leaky_relu(self.trans4(z), 0.2)
        z = F.leaky_relu(self.trans5(z), 0.2)
        z = F.leaky_relu(self.trans6(z), 0.2)
        z = F.leaky_relu(self.trans7(z), 0.2)
        z = torch.sigmoid(z)
        
        return z

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(self.in_channels, 32, 3) #30
        self.conv2 = nn.Conv2d(32, 64, 3)               #28
        self.maxp1 = nn.MaxPool2d(2, 2)                 #14
        self.conv3 = nn.Conv2d(64, 128, 3)              #12
        self.maxp2 = nn.MaxPool2d(2, 2)                 #6
        self.conv4 = nn.Conv2d(128, 256, 3)             #4
        self.maxp3 = nn.MaxPool2d(2, 2)                 #2

        self.fc1 = nn.Linear(256*2*2, 1)
        self.fc2 = nn.Linear(256*2*2, 10)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = self.maxp1(x)
        x = F.leaky_relu(self.conv3(x), 0.1)
        x = self.maxp2(x)
        x = F.leaky_relu(self.conv4(x), 0.1)
        x = self.maxp3(x)
        x = x.view(-1, 256*2*2)
        con = torch.sigmoid(self.fc1(x))
        cla = F.softmax(self.fc2(x), dim=1)

        return con, cla
        
def main():
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    batch_size = 384
    workers = 6
    latent_dim = 128
    lr = 0.001
    num_epochs = 150
    validate_every = 1
    print_every = 100

    save_path = os.path.join(os.path.curdir, "visualize", "gan")
    if not os.path.exists(os.path.join(os.path.curdir, "visualize", "gan")):
        os.makedirs(os.path.join(os.path.curdir, "visualize", "gan"))
    ckpt_path = 'acgan.pt'


    tfms = transforms.Compose([
        transforms.ToTensor(), 
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,
        transform=tfms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=workers,
        drop_last=True)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True,
        transform=tfms)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=workers,
        drop_last=True)


    # -----
    # Model
    generator = Generator(latent_dim=latent_dim).cuda()

    discriminator = Discriminator().cuda()

    # -----
    # Losses

    adv_loss = nn.BCELoss()
    aux_loss = nn.NLLLoss()

    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        adv_loss = adv_loss.cuda()
        aux_loss = aux_loss.cuda()

    # Optimizers for Discriminator and Generator, separate

    optimizer_D = optim.Adam(discriminator.parameters(), lr)
    optimizer_G = optim.Adam(generator.parameters(), lr)


    # -----
    # Train loop

    def denormalize(x):
        """Denomalize a normalized image back to uint8.
        """
        x *= 255
        x = x.transpose(1, 3).transpose(1, 2).cpu().detach().numpy()
        x_denormalized = x.astype(np.uint8)
        return x_denormalized

    # For visualization part
    # Generate 20 random sample for visualization
    # Keep this outside the loop so we will generate near identical images with the same latent featuresper train epoch
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    label = np.random.randint(0,10,batch_size)
    noise = ((torch.from_numpy(noise)).float()).cuda()
    label = ((torch.from_numpy(label)).long()).cuda()
    random_z = noise
    random_y = label


    def train_step(x, y):
        """One train step for AC-GAN.
        You should return loss_g, loss_d, acc_d, a.k.a:
            - average train loss over batch for generator
            - average train loss over batch for discriminator
            - auxiliary train accuracy over batch
        """

        real_label = torch.FloatTensor(batch_size, 1).cuda()
        real_label.fill_(1)

        fake_label = torch.FloatTensor(batch_size, 1).cuda()
        fake_label.fill_(0)

        # y = F.one_hot(y, 10)

        optimizer_D.zero_grad()
        adv, aux = discriminator(x)
        adv_err = adv_loss(adv, real_label)
        aux_err = aux_loss(aux, y)
        loss_1 = adv_err + aux_err
        loss_1.backward()
        optimizer_D.step()

        correct = 0
        aux = aux.data.max(1)[1]
        correct = aux.eq(y.data).cpu().sum()
        acc_d = float(correct) / float(len(y.data))

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        label = np.random.randint(0,10,batch_size)
        noise = ((torch.from_numpy(noise)).float()).cuda()
        label = ((torch.from_numpy(label)).long()).cuda()
        noise_img = generator(noise, y)
        adv2, aux2 = discriminator(noise_img.detach())
        adv2_err = adv_loss(adv2, fake_label)
        aux2_err = aux_loss(aux2, label)
        loss_d = adv2_err + aux2_err
        loss_d.backward()
        optimizer_D.step()

        generator.zero_grad()
        adv3, aux3 = discriminator(noise_img)
        adv3_err = adv_loss(adv3, real_label)
        aux3_err = aux_loss(aux3, y)
        loss_g = adv3_err + aux3_err
        loss_g.backward()
        optimizer_G.step()

        return loss_g.detach().cpu(), loss_d.detach().cpu(), acc_d


    def test(
        test_loader,
        ):
        """Calculate accuracy over Cifar10 test set.
        """
        size = len(test_loader.dataset)
        corrects = 0

        discriminator.eval()
        with torch.no_grad():
            for inputs, gts in test_loader:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    gts = gts.cuda()

                # Forward only
                _, outputs = discriminator(inputs)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == gts.data)

        acc = corrects / size
        print("Test Acc: {:.4f}".format(acc))
        return acc


    g_losses = []
    d_losses = []
    best_acc_test = 0.0

    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()

        avg_loss_g, avg_loss_d = 0.0, 0.0
        for i, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # train step
            loss_g, loss_d, acc_d = train_step(x, y)
            avg_loss_g += loss_g * x.shape[0]
            avg_loss_d += loss_d * x.shape[0]

            # Print
            if i % print_every == 0:
                print("Epoch {}, Iter {}: LossD: {:.6f} LossG: {:.6f}, D_acc {:.6f}".format(epoch, i, loss_g, loss_d, acc_d))

        g_losses.append(avg_loss_g / len(train_dataset))
        d_losses.append(avg_loss_d / len(train_dataset))

        # Save
        if epoch % validate_every == 0:
            acc_test = test(test_loader)
            if acc_test > best_acc_test:
                best_acc_test = acc_test
                # Wrap things to a single dict to train multiple model weights
                state_dict = {
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    }
                torch.save(state_dict, ckpt_path)
                print("Best model saved w/ Test Acc of {:.6f}.".format(best_acc_test))


            # Do some reconstruction
            generator.eval()
            with torch.no_grad():
                # Forward
                xg = generator(random_z, random_y)
                xg = denormalize(xg)

                # Plot 20 randomly generated images
                plt.figure(figsize=(10, 5))
                for p in range(20):
                    plt.subplot(4, 5, p+1)
                    plt.imshow(xg[p])
                    plt.text(0, 0, "{}".format(classes[random_y[p].item()]), color='black',
                                backgroundcolor='white', fontsize=8)
                    plt.axis('off')

                plt.savefig(os.path.join(os.path.join(save_path, "E{:d}.png".format(epoch))), dpi=300)
                plt.clf()
                plt.close('all')

            # Plot losses
            plt.figure(figsize=(10,5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(g_losses, label="G")
            plt.plot(d_losses, label="D")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.xlim([1, epoch])
            plt.legend()
            plt.savefig(os.path.join(os.path.join(save_path, "loss.png")), dpi=300)

    score = compute_score(best_acc_test, 0.65, 0.69)
    print("Your final accuracy:", best_acc_test)

if __name__ == '__main__':
    main()
