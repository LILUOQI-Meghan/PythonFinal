import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir='logs')
# ---------------------------
# 超参数
# ---------------------------
batch_size = 256
epochs = 300             
lr_AKB = 1e-3            # 学习率
lr_E = 2e-3              # Eve 学习率
eve_steps = 5            # 每批训练 Eve 的步数
mu = 1                   # 对抗权重，loss = L_b - mu * L_e
key_priv_dim = 64        # 私钥向量维度  (key_priv_dim,)
key_pub_dim = 64         # 公钥向量维度  (key_pub_dim,)
input_ch = 1
img_h = 28
img_w = 28
input_dim = input_ch * img_h * img_w  # 784 for MNIST
cipher_dim = 64         # 密文向量维度  (B, cipher_dim)

# ---------------------------
# 简单模块定义（尽量小）
# ---------------------------
class KeyGen(nn.Module):
    def __init__(self, priv_dim, pub_dim):
        super().__init__()
        # k_priv 是一个可训练私钥向量 (priv_dim,)
        
        self.net = nn.Sequential(
            nn.Linear(priv_dim, 128),
            nn.ReLU(),
            nn.Linear(128, pub_dim),
        )
    def forward(self,k_priv):
        # 输出 k_pub (pub_dim,)
        return self.net(k_priv)

class Alice(nn.Module):
    def __init__(self, pub_dim, input_dim, cipher_dim):
        super().__init__()
        # 输入: x_flat (B, input_dim), k_pub expanded (B, pub_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + pub_dim, 512),
            nn.ReLU(),
            nn.Linear(512, cipher_dim),
        )
    def forward(self, x_flat, k_pub):
        # x_flat: (B, input_dim)
        # k_pub: (pub_dim,) or (B, pub_dim)
        if k_pub.dim() == 1:
            k = k_pub.unsqueeze(0).expand(x_flat.size(0), -1)  # (B, pub_dim)
        else:
            k = k_pub
        inp = torch.cat([x_flat, k], dim=1)  # (B, input_dim + pub_dim)
        c = self.net(inp)  # (B, cipher_dim) ciphertext
        return c

class Bob(nn.Module):
    def __init__(self, priv_dim, cipher_dim, output_dim):
        super().__init__()
        # 输入: c (B, cipher_dim), k_priv (priv_dim,) expanded -> concat
        self.net = nn.Sequential(
            nn.Linear(cipher_dim + priv_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid(),  # 输出像素在 [0,1]
        )
    def forward(self, c, k_priv):
        # c: (B, cipher_dim), k_priv: (priv_dim,) or (B, priv_dim)
        if k_priv.dim() == 1:
            k = k_priv.unsqueeze(0).expand(c.size(0), -1)  # (B, priv_dim)
        else:
            k = k_priv
        inp = torch.cat([c, k], dim=1)  # (B, cipher_dim + priv_dim)
        x_hat_flat = self.net(inp)  # (B, output_dim)
        return x_hat_flat

class Eve(nn.Module):
    def __init__(self, pub_dim, cipher_dim, output_dim):
        super().__init__()
        # Eve 看到的是: c (B, cipher_dim) + k_pub (public)
        self.net = nn.Sequential(
            nn.Linear(cipher_dim + pub_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid(),
        )
    def forward(self, c, k_pub):
        # c: (B, cipher_dim); k_pub: (pub_dim,) or (B, pub_dim)
        if k_pub.dim() == 1:
            k = k_pub.unsqueeze(0).expand(c.size(0), -1)
        else:
            k = k_pub
        inp = torch.cat([c, k], dim=1)  # (B, cipher_dim + pub_dim)
        x_eve_flat = self.net(inp)  # (B, output_dim)
        return x_eve_flat


transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
val_ds = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader= DataLoader(val_ds, batch_size=batch_size, shuffle=True, drop_last=True)

knet = KeyGen(key_priv_dim, key_pub_dim).to(device)   # k_priv: (key_priv_dim,)
alice = Alice(key_pub_dim, input_dim, cipher_dim).to(device)
bob = Bob(key_priv_dim, cipher_dim, input_dim).to(device)
eve = Eve(key_pub_dim, cipher_dim, input_dim).to(device)


opt_AKB = optim.Adam(list(alice.parameters()) + list(knet.parameters()) + list(bob.parameters()), lr=lr_AKB)
opt_E = optim.Adam(eve.parameters(), lr=lr_E)

mse = nn.MSELoss()

for epoch in range(epochs):
    alice.train(); knet.train(); bob.train(); eve.train()
    epoch_loss_b = 0.0
    epoch_loss_e = 0.0
    for batch_idx, (x, _) in enumerate(train_loader):
        r = (torch.randn(key_priv_dim) * 0.01).to(device)
        x = x.to(device)  # (B,1,28,28)
        B = x.size(0)
        x_flat = x.view(B, -1)  # (B, input_dim)

        # ---------- 1) 训练 Eve---------
        for p in alice.parameters(): p.requires_grad = False
        for p in knet.parameters(): p.requires_grad = False
        for p in bob.parameters(): p.requires_grad = False
        for p in eve.parameters(): p.requires_grad = True

        # 计算公钥并让 Alice 生成 ciphertext，但不需要梯度回 Alice
        with torch.no_grad():
            k_pub = knet(r)  # (key_pub_dim,)
            c = alice(x_flat, k_pub)  # (B, cipher_dim)

        # 多步更新 Eve
        for _ in range(eve_steps):
            opt_E.zero_grad()
            x_eve_flat = eve(c, k_pub)  # (B, input_dim)
            loss_e = mse(x_eve_flat, x_flat)
            loss_e.backward()
            opt_E.step()

        # ---------- 2) 训练  ----------
        for p in alice.parameters(): p.requires_grad = True
        for p in knet.parameters(): p.requires_grad = True
        for p in bob.parameters(): p.requires_grad = True
        for p in eve.parameters(): p.requires_grad = False  # 冻结 Eve 参数

        opt_AKB.zero_grad()
        # 现在 Alice and KeyGen produce c with grad
        k_pub = knet(r)  # (key_pub_dim,)
        c = alice(x_flat, k_pub)  # (B, cipher_dim)
        # Bob 解密
        x_hat_flat = bob(c, r)  # Bob 使用 knet.k_priv (priv_dim,)
        L_b = mse(x_hat_flat, x_flat)

        # Eve 在其固定参数下对当前 c 的恢复（其参数被冻结，但允许梯度回流到 c）
        x_eve_flat = eve(c, k_pub)  # (B, input_dim)
        L_e = mse(x_eve_flat, x_flat)

        # Alice/KeyGen/Bob 的目标：最小化 L_b - mu * L_e
        loss_AKB = L_b - mu * L_e
        loss_AKB.backward()
        opt_AKB.step()
    alice.eval(); bob.eval(); eve.eval(); knet.eval()
    for batch_idx, (x, _) in enumerate(val_loader):
        r = (torch.randn(key_priv_dim) * 0.01).to(device)
        x = x.to(device)  # (B,1,28,28)
        B = x.size(0)
        x_flat = x.view(B, -1)  # (B, input_dim)
        k_pub = knet(r)  # (key_pub_dim,)
        c = alice(x_flat, k_pub)  # (B, cipher_dim)
        # Bob 解密
        x_hat_flat = bob(c, r)  # Bob 使用 knet.k_priv (priv_dim,)
        L_b = mse(x_hat_flat, x_flat)

        # Eve 在其固定参数下对当前 c 的恢复（其参数被冻结，但允许梯度回流到 c）
        x_eve_flat = eve(c, k_pub)  # (B, input_dim)
        L_e = mse(x_eve_flat, x_flat)

        # Alice/KeyGen/Bob 的目标：最小化 L_b - mu * L_e
        loss_AKB = L_b - mu * L_e


        epoch_loss_b += L_b.item()
        epoch_loss_e += L_e.item()
        # 简短输出以快速查看训练进度（每 200 batches）
        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch {epoch+1} Batch {batch_idx+1} | L_b {L_b.item():.4f} L_e {L_e.item():.4f}")

    # 每 epoch 简要打印平均损失
    n_batches = len(train_loader)
    print(f"Epoch {epoch+1} finished. avg L_b: {epoch_loss_b/n_batches:.4f}, avg L_e: {epoch_loss_e/n_batches:.4f}")
    writer.add_scalar('Loss/L_b', epoch_loss_b/n_batches, epoch)
    writer.add_scalar('Loss/L_e', epoch_loss_e/n_batches, epoch)
    writer.add_scalar('Loss/L_alice_kpub', epoch_loss_b/n_batches - mu * epoch_loss_e/n_batches, epoch)
    writer.add_images('Images/Original', x[:4,:,:,:].cpu(), epoch)
    writer.add_images('Images/Reconstructed', x_hat_flat.view(B,1,28,28)[:4,:,:,:].detach().cpu(), epoch)
    writer.add_images('Images/Eve', x_eve_flat.view(B,1,28,28)[:4,:,:,:].detach().cpu(), epoch)

alice.eval(); bob.eval(); eve.eval(); knet.eval()
r = (torch.randn(key_priv_dim) * 0.01).to(device)
x, _ = next(iter(train_loader))
x = x.to(device)
B = x.size(0)
x_flat = x.view(B, -1)
k_pub = knet(r)  # (key_pub_dim,)
c = alice(x_flat, k_pub)  # (B, cipher_dim)
x_hat = bob(c, r).view(B, 1, 28, 28)  # (B,1,28,28)
x_eve = eve(c, k_pub).view(B, 1, 28, 28)


import torchvision.utils as vutils
sample = torch.cat([x[:16].cpu(), x_hat[:16].detach().cpu(), x_eve[:16].detach().cpu()], dim=0)
vutils.save_image(sample, "sample_results.png", nrow=8)


torch.save({
    'knet': knet.state_dict(),
    'alice': alice.state_dict(),
    'bob': bob.state_dict(),
    'eve': eve.state_dict(),
}, "models_minimal.pth")

print("Done. sample_results.png and models_minimal.pth saved.")
