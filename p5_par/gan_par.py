# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# # Data loading and preprocessing

# %%

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=128, shuffle=True)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)

# %% [markdown]
# # Models definitions

# %% [markdown]
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

# %% [markdown]
# ## Discriminator (CNN-based from P3)
# %%


class CNN_ParImpar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 16, kernel_size=4),
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(
            nn.Linear(16*9*9, 2),  # 2 clases: par (0) e impar (1)
        )

    def forward(self, x):
        # x: [batch, 784] -> reshape to [batch, 1, 28, 28]
        x = x.view(-1, 1, 28, 28)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class Discriminator(nn.Module):
    def __init__(self, classifier: CNN_ParImpar, target_class: int = 0):
        super().__init__()
        self.classifier = classifier
        self.target_class = target_class  # 0=par, 1=impar

    def forward(self, x):
        # x: [batch, 784]
        logits = self.classifier(x)  # [batch, 2]
        probs = torch.softmax(logits, dim=1)  # [batch, 2]
        # Retornar la probabilidad de la clase objetivo
        return probs[:, self.target_class].unsqueeze(1)  # [batch, 1]

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


def verify_classifier(classifier, test_loader, device, n_samples=10):
    print("\n🔍 Verificando clasificador...")
    classifier.eval()

    test_images, test_labels = next(iter(test_loader))
    test_images = test_images[:n_samples].to(device)
    test_labels = test_labels[:n_samples].to(device)
    test_labels_par_impar = test_labels % 2

    with torch.no_grad():
        outputs = classifier(test_images.view(-1, 28*28))
        _, predicted = torch.max(outputs, 1)

    correct = (predicted == test_labels_par_impar).sum().item()

    print(f"  Predicciones: {predicted.cpu().numpy()}")
    print(f"  Etiquetas reales (par/impar): {test_labels_par_impar.cpu().numpy()}")
    print(f"  Aciertos: {correct}/{n_samples}")
    print(f"✅ Clasificador verificado (accuracy: {100*correct/n_samples:.1f}%)\n")

    return correct / n_samples


def evaluate_gan(generator, discriminator, test_loader, criterion, fixed_noise, z_dim, epoch, device):
    print(f"\n{'='*60}")
    print(f"📊 Evaluating at Epoch {epoch}")
    print(f"{'='*60}")

    generate_and_plot_images(generator, fixed_noise, epoch)
    test_metrics = evaluate_on_test_set(generator, discriminator, test_loader, criterion, z_dim, device)

    print(f"\n📈 Test Set Evaluation:")
    print(f"  • Discriminator loss: {test_metrics['loss_disc']:.4f}")
    print(f"  • Generator loss: {test_metrics['loss_gen']:.4f}")
    print(f"{'='*60}\n")

    return test_metrics

# %% [markdown]
# # Training

# %%


def train_disc(discriminator, generator, real, criterion, optimizer_disc, z_dim, device):
    batch_size = real.size(0)

    label_real = torch.ones(batch_size, 1, device=device)
    label_fake = torch.zeros(batch_size, 1, device=device)

    noise = torch.randn(batch_size, z_dim, device=device)
    noise = torch.clamp(noise, -2.0, 2.0)
    fake = generator(noise)

    # Entrenar discriminador
    loss_disc_real = criterion(discriminator(real), label_real)
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

# Cargar el generador preentrenado
print("\n" + "="*60)
print("📂 Cargando modelos preentrenados...")
print("="*60)

generator = Generator(z_dim).to(device)
checkpoint_gen = torch.load('best_gan_model.pth', map_location=device)
generator.load_state_dict(checkpoint_gen['generator_state_dict'])
print("✅ Generador cargado desde: best_gan_model.pth")

# Cargar el clasificador par/impar preentrenado
classifier_par_impar = CNN_ParImpar().to(device)
classifier_par_impar.load_state_dict(torch.load('model_odd_even.pth', map_location=device))
print("✅ Clasificador Par/Impar cargado desde: model_odd_even.pth")

# Crear discriminador usando el clasificador
target_class = 0  # 0 = pares, 1 = impares
discriminator = Discriminator(classifier_par_impar, target_class=target_class).to(device)
print(f"✅ Discriminador creado para generar: {'PARES' if target_class == 0 else 'IMPARES'}")
print("="*60 + "\n")

lr_disc = 0.0002
lr_gen = 0.0002

criterion = nn.BCELoss()

# Entrenar generador y discriminador
optimizer_gen = optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))

# Fixed noise for consistent visualization across epochs
fixed_noise = torch.randn(16, z_dim, device=device)
fixed_noise = torch.clamp(fixed_noise, -2.0, 2.0)

# Verificar que los modelos cargados funcionan correctamente
print("\n" + "="*60)
print("🔍 VERIFICACIÓN DE MODELOS PREENTRENADOS")
print("="*60)

# 1. Verificar generador
print("\n🎨 1. Generador - Imágenes iniciales:")
generate_and_plot_images(generator, fixed_noise, 0)

# 2. Verificar clasificador
print("🧠 2. Clasificador Par/Impar:")
verify_classifier(classifier_par_impar, test_loader, device, n_samples=10)


num_epochs = 120
n_gen_iterations = 1

print(f"🎯 Objetivo: Generar dígitos {'PARES' if target_class == 0 else 'IMPARES'}")
print(f"📋 Épocas de entrenamiento: {num_epochs}\n")

losses_disc = []
losses_gen = []
eval_epochs = []
eval_losses_gen = []
eval_losses_disc = []

best_loss_gen = float('inf')

for epoch in range(num_epochs):
    generator.train()
    discriminator.train()

    epoch_loss_g = 0
    epoch_loss_d = 0
    num_batches = 0
    for real, _ in dataloader:
        real = real.view(-1, 28*28).to(device)

        # Train Discriminator
        loss_disc = train_disc(discriminator, generator, real, criterion, optimizer_disc, z_dim, device)

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

    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
        test_metrics = evaluate_gan(generator, discriminator, test_loader, criterion, fixed_noise, z_dim, epoch, device)

        eval_epochs.append(epoch)
        eval_losses_gen.append(test_metrics['loss_gen'])
        eval_losses_disc.append(test_metrics['loss_disc'])

        if test_metrics['loss_gen'] < best_loss_gen:
            best_loss_gen = test_metrics['loss_gen']
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_gen_state_dict': optimizer_gen.state_dict(),
                'optimizer_disc_state_dict': optimizer_disc.state_dict(),
                'loss_gen': avg_loss_g,
                'loss_disc': avg_loss_d,
            }, 'gan_par_model.pth')
            print(f"New best model saved!")

            # También guardar el clasificador por separado para usarlo después
            torch.save(classifier_par_impar.state_dict(), 'discriminator_trained.pth')
            print(f"Discriminator (classifier) saved to: discriminator_trained.pth")

# %% [markdown]
# # Save metrics to CSV

# %%


eval_metrics = {
    'epoch': eval_epochs,
    'loss_disc': eval_losses_disc,
    'loss_gen': eval_losses_gen
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
