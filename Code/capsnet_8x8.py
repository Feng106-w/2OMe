# models/capsnet_8x8.py 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

SIDE = 8 

class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=3):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=1)
    def forward(self, x):
        return F.relu(self.conv(x))

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=8, kernel_size=3):
        super(PrimaryCaps, self).__init__()
        self.num_capsules = num_capsules
        self.out_channels  = out_channels  # = 8
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=1, padding=1)
            for _ in range(num_capsules)
        ])
    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]                 # list of [B,8,SIDE,SIDE]
        u = torch.stack(u, dim=1)                                      # [B,num_caps,8,SIDE,SIDE]
        u = u.view(x.size(0), self.num_capsules * SIDE * SIDE, -1)     # [B, num_routes, 8]
        return self.squash(u)
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        return squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm + 1e-9))

class DigitCaps(nn.Module):
    def __init__(self, num_capsules=2, num_routes=None, in_channels=8, out_channels=32):
        super(DigitCaps, self).__init__()
        if num_routes is None:
            raise ValueError("DigitCaps requires explicit num_routes = Primary_capsule_num * SIDE * SIDE")
        self.in_channels   = in_channels     
        self.num_routes    = num_routes
        self.num_capsules  = num_capsules
        self.out_channels  = out_channels
        # W: [1, num_routes, num_capsules, out_ch, in_ch]
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):  # x: [B, num_routes, 8]
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA: b_ij = b_ij.cuda()

        num_iterations = 3
        c_last = None  
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)                       
            c_last = c_ij                                
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)  # [B, num_routes, num_caps, 1, 1]
            s_j  = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j  = self.squash(s_j)
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        # c_last: [1, num_routes, num_caps, 1] -> [B, num_routes, num_caps]
        c_batch = torch.cat([c_last] * batch_size, dim=0).squeeze(-1)  # [B, num_routes, 2]

        # num_routes = (num_primary=8) * SIDE * SIDE = 8 * 64
        num_primary = 8
        c_primary = c_batch.view(batch_size, num_primary, SIDE*SIDE, self.num_capsules).mean(dim=2)  # [B, 8, 2]
        self.last_c_primary = c_primary
        
        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-2, keepdim=True)
        return squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm + 1e-9))

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(32 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1 * SIDE * SIDE),
            nn.Sigmoid()
        )
    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes, dim=1)  
        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.eye(2))
        if USE_CUDA: masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
        reconstructions = self.reconstraction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
        reconstructions = reconstructions.view(-1, 1, SIDE, SIDE)
        return reconstructions, masked

class CapsNet(nn.Module):
    def __init__(self, Primary_capsule_num=8):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps(num_capsules=Primary_capsule_num)
        self.digit_capsules = DigitCaps(
            in_channels=8,
            num_routes=Primary_capsule_num * SIDE * SIDE,
            num_capsules=2,
            out_channels=32
        )
        self.decoder = Decoder()
        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)
        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))
        left  = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)
        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()
        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1),
                             data.view(reconstructions.size(0), -1))
        return loss * 0.0001
