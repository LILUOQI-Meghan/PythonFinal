import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir='logs')

batch_size = 256
epochs = 300            
lr_AKB = 1e-3            # Alice/KeyGen/Bob 学习率
lr_E = 2e-3              # Eve 学习率
eve_steps = 5            # 每批训练 Eve 的步数
mu = 1                 # 用于 Alice 的对抗权重，loss = L_b - mu * L_e
random_dim = 64          # 随机向量维度
key_priv_dim = 64        # 私钥向量维度  (key_priv_dim,)
key_pub_dim = 64         # 公钥向量维度  (key_pub_dim,)
input_ch = 1
img_h = 28
img_w = 28
input_dim = input_ch * img_h * img_w  # 784 for MNIST
cipher_dim = 64         # 密文向量维度  (B, cipher_dim)

class PrivGen(nn.Module):
    def __init__(self, random_dim, priv_dim):
        super().__init__()
        # 从随机向量生成私钥 - 加深网络
        self.net = nn.Sequential(
            nn.Linear(random_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, priv_dim),
            nn.Tanh()
        )
    
    def forward(self, random_vec):
        # 输入: random_vec (random_dim,)
        # 输出: private_key (priv_dim,)
        return self.net(random_vec)

class PubGen(nn.Module):
    def __init__(self, random_dim, priv_dim, pub_dim):
        super().__init__()
        # 从随机向量和私钥生成公钥 - 加深网络
        self.net = nn.Sequential(
            nn.Linear(random_dim + priv_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, pub_dim),
            nn.Tanh()
        )
    
    def forward(self, random_vec, private_key):

        combined = torch.cat([random_vec, private_key], dim=0)
        return self.net(combined)

class Alice(nn.Module):
    def __init__(self, pub_dim, input_dim, cipher_dim):
        super().__init__()
        # 加深网络，每层都输入公钥
        self.fc1 = nn.Linear(input_dim + pub_dim, 512)
        self.fc2 = nn.Linear(512 + pub_dim, 512)
        self.fc3 = nn.Linear(512 + pub_dim, 256)
        self.fc4 = nn.Linear(256 + pub_dim, cipher_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x_flat, k_pub):
        # x_flat: (B, input_dim)
        # k_pub: (pub_dim,) or (B, pub_dim)
        if k_pub.dim() == 1:
            k = k_pub.unsqueeze(0).expand(x_flat.size(0), -1)  # (B, pub_dim)
        else:
            k = k_pub
        
        # 每层都输入公钥
        x1 = torch.cat([x_flat, k], dim=1)  # (B, input_dim + pub_dim)
        x2 = self.relu(self.fc1(x1))
        x3 = torch.cat([x2, k], dim=1)  # (B, 512 + pub_dim)
        x4 = self.relu(self.fc2(x3))
        x5 = torch.cat([x4, k], dim=1)  # (B, 512 + pub_dim)
        x6 = self.relu(self.fc3(x5))
        x7 = torch.cat([x6, k], dim=1)  # (B, 256 + pub_dim)
        c = self.fc4(x7)  # (B, cipher_dim) ciphertext
        return c

class Bob(nn.Module):
    def __init__(self, priv_dim, cipher_dim, output_dim):
        super().__init__()
        # 加深网络，每层都输入私钥
        self.fc1 = nn.Linear(cipher_dim + priv_dim, 512)
        self.fc2 = nn.Linear(512 + priv_dim, 512)
        self.fc3 = nn.Linear(512 + priv_dim, 256)
        self.fc4 = nn.Linear(256 + priv_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # 输出像素在 [0,1]
        
    def forward(self, c, k_priv):
        # c: (B, cipher_dim), k_priv: (priv_dim,) or (B, priv_dim)
        if k_priv.dim() == 1:
            k = k_priv.unsqueeze(0).expand(c.size(0), -1)  # (B, priv_dim)
        else:
            k = k_priv
        
        # 每层都输入私钥
        x1 = torch.cat([c, k], dim=1)  # (B, cipher_dim + priv_dim)
        x2 = self.relu(self.fc1(x1))
        x3 = torch.cat([x2, k], dim=1)  # (B, 512 + priv_dim)
        x4 = self.relu(self.fc2(x3))
        x5 = torch.cat([x4, k], dim=1)  # (B, 512 + priv_dim)
        x6 = self.relu(self.fc3(x5))
        x7 = torch.cat([x6, k], dim=1)  # (B, 256 + priv_dim)
        x8 = self.fc4(x7)
        x_hat_flat = self.sigmoid(x8)  # (B, output_dim)
        return x_hat_flat

class Eve(nn.Module):
    def __init__(self, pub_dim, cipher_dim, output_dim):
        super().__init__()
        # 加深网络，每层都输入公钥
        self.fc1 = nn.Linear(cipher_dim + pub_dim, 512)
        self.fc2 = nn.Linear(512 + pub_dim, 512)
        self.fc3 = nn.Linear(512 + pub_dim, 256)
        self.fc4 = nn.Linear(256 + pub_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, c, k_pub):
        # c: (B, cipher_dim); k_pub: (pub_dim,) or (B, pub_dim)
        if k_pub.dim() == 1:
            k = k_pub.unsqueeze(0).expand(c.size(0), -1)
        else:
            k = k_pub
        
        # 每层都输入公钥
        x1 = torch.cat([c, k], dim=1)  # (B, cipher_dim + pub_dim)
        x2 = self.relu(self.fc1(x1))
        x3 = torch.cat([x2, k], dim=1)  # (B, 512 + pub_dim)
        x4 = self.relu(self.fc2(x3))
        x5 = torch.cat([x4, k], dim=1)  # (B, 512 + pub_dim)
        x6 = self.relu(self.fc3(x5))
        x7 = torch.cat([x6, k], dim=1)  # (B, 256 + pub_dim)
        x8 = self.fc4(x7)
        x_eve_flat = self.sigmoid(x8)  # (B, output_dim)
        return x_eve_flat


transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
val_ds = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader= DataLoader(val_ds, batch_size=batch_size, shuffle=True, drop_last=True)

priv_gen = PrivGen(random_dim, key_priv_dim).to(device)   # 从随机向量生成私钥
pub_gen = PubGen(random_dim, key_priv_dim, key_pub_dim).to(device)   # 从随机向量和私钥生成公钥
alice = Alice(key_pub_dim, input_dim, cipher_dim).to(device)
bob = Bob(key_priv_dim, cipher_dim, input_dim).to(device)
eve = Eve(key_pub_dim, cipher_dim, input_dim).to(device)

opt_AKB = optim.Adam(list(alice.parameters()) + list(priv_gen.parameters()) + list(pub_gen.parameters()) + list(bob.parameters()), lr=lr_AKB)
opt_E = optim.Adam(eve.parameters(), lr=lr_E)

mse = nn.MSELoss()


for epoch in range(epochs):
    alice.train(); priv_gen.train(); pub_gen.train(); bob.train(); eve.train()
    epoch_loss_b = 0.0
    epoch_loss_e = 0.0
    for batch_idx, (x, _) in enumerate(train_loader):
        r = (torch.randn(random_dim) * 0.01).to(device)
        x = x.to(device)  # (B,1,28,28)
        B = x.size(0)
        x_flat = x.view(B, -1)  # (B, input_dim)

        # ---------- 1) 训练 Eve（Alice/KeyGen/Bob 固定） ----------
        # 冻结 Alice/KeyGen/Bob 参数
        for p in alice.parameters(): p.requires_grad = False
        for p in priv_gen.parameters(): p.requires_grad = False
        for p in pub_gen.parameters(): p.requires_grad = False
        for p in bob.parameters(): p.requires_grad = False
        for p in eve.parameters(): p.requires_grad = True

        # 计算公钥并让 Alice 生成 ciphertext，但不需要梯度回 Alice
        with torch.no_grad():
            k_pub = pub_gen(r, priv_gen(r))  # (key_pub_dim,)
            c = alice(x_flat, k_pub)  # (B, cipher_dim)

        # 多步更新 Eve
        for _ in range(eve_steps):
            opt_E.zero_grad()
            x_eve_flat = eve(c, k_pub)  # (B, input_dim)
            loss_e = mse(x_eve_flat, x_flat)
            loss_e.backward()
            opt_E.step()

        # ---------- 2) 训练 Alice/KeyGen/Bob（Eve 固定） ----------
        for p in alice.parameters(): p.requires_grad = True
        for p in priv_gen.parameters(): p.requires_grad = True
        for p in pub_gen.parameters(): p.requires_grad = True
        for p in bob.parameters(): p.requires_grad = True
        for p in eve.parameters(): p.requires_grad = False  # 冻结 Eve 参数

        opt_AKB.zero_grad()
        # 现在 Alice and KeyGen produce c with grad
        k_pub = pub_gen(r, priv_gen(r))  # (key_pub_dim,)
        c = alice(x_flat, k_pub)  # (B, cipher_dim)
        # Bob 解密
        x_hat_flat = bob(c, priv_gen(r))  # Bob 使用 priv_gen.k_priv (priv_dim,)
        L_b = mse(x_hat_flat, x_flat)

        # Eve 在其固定参数下对当前 c 的恢复（其参数被冻结，但允许梯度回流到 c）
        x_eve_flat = eve(c, k_pub)  # (B, input_dim)
        L_e = mse(x_eve_flat, x_flat)

        # Alice/KeyGen/Bob 的目标：最小化 L_b - mu * L_e
        loss_AKB = L_b - mu * L_e
        loss_AKB.backward()
        opt_AKB.step()
    alice.eval(); bob.eval(); eve.eval(); priv_gen.eval(); pub_gen.eval()
    for batch_idx, (x, _) in enumerate(val_loader):
        r = (torch.randn(random_dim) * 0.01).to(device)
        x = x.to(device)  # (B,1,28,28)
        B = x.size(0)
        x_flat = x.view(B, -1)  # (B, input_dim)
        k_pub = pub_gen(r, priv_gen(r))  # (key_pub_dim,)
        c = alice(x_flat, k_pub)  # (B, cipher_dim)
        # Bob 解密
        x_hat_flat = bob(c, priv_gen(r))  # Bob 使用 priv_gen.k_priv (priv_dim,)
        L_b = mse(x_hat_flat, x_flat)

        # Eve 在其固定参数下对当前 c 的恢复（其参数被冻结，但允许梯度回流到 c）
        x_eve_flat = eve(c, k_pub)  # (B, input_dim)
        L_e = mse(x_eve_flat, x_flat)

        # Alice/KeyGen/Bob 的目标：最小化 L_b - mu * L_e
        loss_AKB = L_b - mu * L_e


        epoch_loss_b += L_b.item()
        epoch_loss_e += L_e.item()
        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch {epoch+1} Batch {batch_idx+1} | L_b {L_b.item():.4f} L_e {L_e.item():.4f}")

    n_batches = len(train_loader)
    print(f"Epoch {epoch+1} finished. avg L_b: {epoch_loss_b/n_batches:.4f}, avg L_e: {epoch_loss_e/n_batches:.4f}")
    writer.add_scalar('Loss/L_b', epoch_loss_b/n_batches, epoch)
    writer.add_scalar('Loss/L_e', epoch_loss_e/n_batches, epoch)
    writer.add_scalar('Loss/L_alice_kpub', epoch_loss_b/n_batches - mu * epoch_loss_e/n_batches, epoch)
    writer.add_images('Images/Original', x[:4,:,:,:].cpu(), epoch)
    writer.add_images('Images/Reconstructed', x_hat_flat.view(B,1,28,28)[:4,:,:,:].detach().cpu(), epoch)
    writer.add_images('Images/Eve', x_eve_flat.view(B,1,28,28)[:4,:,:,:].detach().cpu(), epoch)

alice.eval(); bob.eval(); eve.eval(); priv_gen.eval(); pub_gen.eval()
r = (torch.randn(random_dim) * 0.01).to(device)
x, _ = next(iter(train_loader))
x = x.to(device)
B = x.size(0)
x_flat = x.view(B, -1)
k_pub = pub_gen(r, priv_gen(r))  # (key_pub_dim,)
c = alice(x_flat, k_pub)  # (B, cipher_dim)
x_hat = bob(c, priv_gen(r)).view(B, 1, 28, 28)  # (B,1,28,28)
x_eve = eve(c, k_pub).view(B, 1, 28, 28)


import torchvision.utils as vutils
sample = torch.cat([x[:16].cpu(), x_hat[:16].detach().cpu(), x_eve[:16].detach().cpu()], dim=0)
vutils.save_image(sample, "sample_results.png", nrow=8)

torch.save({
    'priv_gen': priv_gen.state_dict(),
    'pub_gen': pub_gen.state_dict(),
    'alice': alice.state_dict(),
    'bob': bob.state_dict(),
    'eve': eve.state_dict(),
}, "models_minimal.pth")

print("Done. sample_results.png and models_minimal.pth saved.")
