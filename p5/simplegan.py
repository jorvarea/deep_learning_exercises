# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import torch.nn.functional as F
import pandas as pd

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% [markdown]
# # Data loading and preprocessing

#%%

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=128, shuffle=True)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)

#%% [markdown]
# # Models definitions

#%% [markdown]
# ## Generator
# %%

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, img_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

#%% [markdown]
# ## Discriminator
# %%

class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# %% [markdown]
# # Define functions to evaluate the models during training

# %%

def generate_and_plot_images(generator, fixed_noise, epoch):
    generator.eval()
    with torch.no_grad():
        gen_imgs = generator(fixed_noise).view(-1, 1, 28, 28)
        gen_imgs = gen_imgs.cpu().numpy()
        
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        axes = axes.flatten()
        for i, (img, ax) in enumerate(zip(gen_imgs, axes)):
            ax.imshow((img.squeeze() + 1) / 2, cmap='gray')
            ax.axis('off')
            ax.set_title(f'Img {i+1}', fontsize=8)
        plt.suptitle(f'Generated Images - Epoch {epoch}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


def evaluate_on_test_set(generator, discriminator, test_loader, criterion, z_dim, device):
    generator.eval()
    discriminator.eval()
    
    test_loss_disc = 0
    test_loss_gen = 0
    test_batches = 0
    
    with torch.no_grad():
        for real_test, _ in test_loader:
            real_test = real_test.view(-1, 28*28).to(device)
            batch_size_test = real_test.size(0)
            
            label_real_test = torch.ones(batch_size_test, 1, device=device)
            label_fake_test = torch.zeros(batch_size_test, 1, device=device)
            
            # Generate fake images
            noise_test = torch.randn(batch_size_test, z_dim, device=device)
            fake_test = generator(noise_test)
            
            # Calculate Discriminator loss
            loss_disc_real = criterion(discriminator(real_test), label_real_test)
            loss_disc_fake = criterion(discriminator(fake_test), label_fake_test)
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            
            # Calculate Generator loss
            output = discriminator(fake_test)
            loss_gen = criterion(output, label_real_test)
            
            test_loss_disc += loss_disc.item()
            test_loss_gen += loss_gen.item()
            test_batches += 1
    
    avg_test_loss_disc = test_loss_disc / test_batches
    avg_test_loss_gen = test_loss_gen / test_batches
    
    return {
        'loss_disc': avg_test_loss_disc,
        'loss_gen': avg_test_loss_gen
    }


def calculate_fid(generator, fid_metric, real_loader, z_dim, device, n_samples=100):
    generator.eval()
    fid_metric.reset()
    
    real_list = []
    with torch.no_grad():
        for real, _ in real_loader:
            real_rgb = real.repeat(1, 3, 1, 1)
            real_299 = F.interpolate(real_rgb, size=(299, 299), mode='bilinear')
            real_299 = ((real_299 + 1) / 2 * 255).clamp(0, 255).byte()
            real_list.append(real_299)
            if len(real_list) * real.size(0) >= n_samples:
                break
    
    real_images = torch.cat(real_list, dim=0)[:n_samples]
    fid_metric.update(real_images, real=True)
    
    batch_size = 50
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            bs = min(batch_size, n_samples - i)
            noise = torch.randn(bs, z_dim, device=device)
            fake = generator(noise).view(-1, 1, 28, 28)
            fake_rgb = fake.repeat(1, 3, 1, 1)
            fake_299 = F.interpolate(fake_rgb, size=(299, 299), mode='bilinear')
            fake_299 = ((fake_299 + 1) / 2 * 255).clamp(0, 255).byte().cpu()
            fid_metric.update(fake_299, real=False)
            
            del noise, fake, fake_rgb, fake_299
    
    return fid_metric.compute().item()


def evaluate_gan(generator, discriminator, fid_metric, test_loader, criterion, fixed_noise, z_dim, epoch, device):
    print(f"\n{'='*60}")
    print(f"📊 Evaluating at Epoch {epoch}")
    print(f"{'='*60}")
    
    generate_and_plot_images(generator, fixed_noise, epoch)    
    test_metrics = evaluate_on_test_set(generator, discriminator, test_loader, criterion, z_dim, device)
    fid = calculate_fid(generator, fid_metric, test_loader, z_dim, device)
    
    print(f"\n📈 Test Set Evaluation:")
    print(f"  • Discriminator loss: {test_metrics['loss_disc']:.4f}")
    print(f"  • Generator loss: {test_metrics['loss_gen']:.4f}")
    print(f"  • FID: {fid:.2f}")
    print(f"{'='*60}\n")
    
    return test_metrics, fid

# %% [markdown]
# # Training

# %%

def train_disc(discriminator, generator, real, criterion, optimizer_disc, z_dim, device, curr_epoch, total_epochs):
    batch_size = real.size(0)
    
    label_real = torch.ones(batch_size, 1, device=device) * 0.9
    label_fake = torch.zeros(batch_size, 1, device=device)

    noise_strength = 0.1 * np.exp(-2.0 * curr_epoch / total_epochs)
    real_noisy = real + noise_strength * torch.randn_like(real)
    
    noise = torch.randn(batch_size, z_dim, device=device)
    noise = torch.clamp(noise, -2.0, 2.0)
    fake = generator(noise)
    
    loss_disc_real = criterion(discriminator(real_noisy), label_real)
    loss_disc_fake = criterion(discriminator(fake.detach()), label_fake)
    loss_disc = (loss_disc_real + loss_disc_fake) / 2

    optimizer_disc.zero_grad()
    loss_disc.backward()
    optimizer_disc.step()

    return loss_disc.item()


def train_gen(generator, discriminator, real, criterion, optimizer_gen, z_dim, device):
    batch_size = real.size(0)
    label_real = torch.ones(batch_size, 1, device=device)
    
    noise = torch.randn(batch_size, z_dim, device=device)
    noise = torch.clamp(noise, -2.0, 2.0)
    fake = generator(noise)
    
    output = discriminator(fake)
    loss_gen = criterion(output, label_real)

    optimizer_gen.zero_grad()
    loss_gen.backward()
    optimizer_gen.step()

    return loss_gen.item()

# %%

z_dim = 100
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)
fid_metric_global = FrechetInceptionDistance(feature=2048, normalize=True)
lr_disc = 0.0002
lr_gen = 0.0002

criterion = nn.BCELoss()
optimizer_gen = optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))

# Fixed noise for consistent visualization across epochs
fixed_noise = torch.randn(16, z_dim, device=device)
fixed_noise = torch.clamp(fixed_noise, -2.0, 2.0)

num_epochs = 120
n_gen_iterations = 1

losses_disc = []
losses_gen = []
fid_values = []
eval_epochs = []
eval_losses_gen = []
eval_losses_disc = []

best_fid = float('inf')

for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    
    epoch_loss_g = 0
    epoch_loss_d = 0
    num_batches = 0
    for real, _ in dataloader:
        real = real.view(-1, 28*28).to(device)

        # Train Discriminator
        loss_disc = train_disc(discriminator, generator, real, criterion,
                              optimizer_disc, z_dim, device, epoch, num_epochs)

        # Train Generator
        total_loss_gen = 0
        for _ in range(n_gen_iterations):
            loss_gen_iter = train_gen(generator, discriminator, real, criterion,
                                optimizer_gen, z_dim, device)
            total_loss_gen += loss_gen_iter

        # Accumulate losses
        epoch_loss_d += loss_disc
        epoch_loss_g += total_loss_gen / n_gen_iterations
        num_batches += 1

    # Calculate the average loss for the epoch and save it
    avg_loss_g = epoch_loss_g / num_batches
    avg_loss_d = epoch_loss_d / num_batches
    losses_gen.append(avg_loss_g)
    losses_disc.append(avg_loss_d)

    print(f"Epoch {epoch}: Loss_D {avg_loss_d}, Loss_G {avg_loss_g}")

    if (epoch + 1) % 10 == 0:
        test_metrics, fid = evaluate_gan(generator, discriminator, fid_metric_global, test_loader, criterion, fixed_noise, z_dim, epoch, device)

        eval_epochs.append(epoch)
        fid_values.append(fid)
        eval_losses_gen.append(test_metrics['loss_gen'])
        eval_losses_disc.append(test_metrics['loss_disc'])


        if fid < best_fid:
            best_fid = fid
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_gen_state_dict': optimizer_gen.state_dict(),
                'optimizer_disc_state_dict': optimizer_disc.state_dict(),
                'fid': fid,
                'loss_gen': avg_loss_g,
                'loss_disc': avg_loss_d,
            }, 'best_gan_model.pth')
            print(f"New best model saved! FID: {fid:.2f}")

# %% [markdown]
# # Save metrics to CSV

# %%

import os

eval_metrics = {
    'epoch': eval_epochs,
    'loss_disc': eval_losses_disc,
    'loss_gen': eval_losses_gen,
    'fid': fid_values
}

training_metrics = {
    'epoch': list(range(num_epochs)),
    'loss_disc': losses_disc,
    'loss_gen': losses_gen,
}

os.makedirs('training_metrics', exist_ok=True)

df_training_metrics = pd.DataFrame(training_metrics)
df_training_metrics.to_csv('training_metrics/training_metrics.csv', index=False)

df_eval_metrics = pd.DataFrame(eval_metrics)
df_eval_metrics.to_csv('training_metrics/eval_metrics.csv', index=False)

# %% [markdown]
# # Plot losses

# %%

plt.figure(figsize=(10, 5))
plt.plot(losses_disc, label='Discriminator Loss', linewidth=2)
plt.plot(losses_gen, label='Generator Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Losses')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
plt.figure(figsize=(10, 5))
epochs_fid = list(range(5, num_epochs+1, 5))[:len(fid_values)]
plt.plot(epochs_fid, fid_values, marker='o', linewidth=2, color='green')
plt.xlabel('Epoch')
plt.ylabel('FID Score')
plt.title('FID Evolution (Lower is Better)')
plt.grid(True, alpha=0.3)
plt.axhline(y=min(fid_values), color='r', linestyle='--', label=f'Best: {min(fid_values):.2f}')
plt.legend()
plt.show()

# %% [markdown]
# # Calculating Inception Score (IS) and Frèchet Inception Distance (FID)
# ## Create real and fake images
# %%

n_samples = 50
noise = torch.randn(n_samples, z_dim, device=device)
with torch.no_grad():
    gen_images = generator(noise).view(-1, 1, 28, 28)

real_images = []
for real, _ in dataloader:
    real_images.append(real)
    if len(real_images) * real.size(0) >= n_samples:
        break
real_images = torch.cat(real_images, dim=0)[:n_samples].to(device)

# Convert from grayscale to RGB
gen_images_rgb = gen_images.repeat(1, 3, 1, 1)
real_images_rgb = real_images.repeat(1, 3, 1, 1)

# Resize to 299x299 (size that Inception needs)
gen_images_299 = F.interpolate(gen_images_rgb, size=(299, 299), mode='bilinear')
real_images_299 = F.interpolate(real_images_rgb, size=(299, 299), mode='bilinear')

# Normalize to [0, 1] since they are in [-1, 1]
gen_images_299 = (gen_images_299 + 1) / 2
real_images_299 = (real_images_299 + 1) / 2

# Convert to uint8 [0, 255]
gen_images_299 = (gen_images_299 * 255).byte()
real_images_299 = (real_images_299 * 255).byte()

# %% [markdown]
# ## Calculate Inception Score and FID using torchmetrics

# %%

is_metric = InceptionScore().to(device)
is_score, is_std = is_metric(gen_images_299)  # generated_images: [N, 3, 299, 299] tensor
print('Inception Score:', is_score.item())
