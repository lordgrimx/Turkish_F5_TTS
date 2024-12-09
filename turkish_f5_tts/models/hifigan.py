import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                     padding=self.get_padding(kernel_size, dilation[0])),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                     padding=self.get_padding(kernel_size, dilation[1])),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                     padding=self.get_padding(kernel_size, dilation[2]))
        ])
        
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                     padding=self.get_padding(kernel_size, 1)),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                     padding=self.get_padding(kernel_size, 1)),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                     padding=self.get_padding(kernel_size, 1))
        ])

    def get_padding(self, kernel_size, dilation=1):
        return int((kernel_size*dilation - dilation)/2)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

class HiFiGAN(nn.Module):
    def __init__(self):
        super(HiFiGAN, self).__init__()
        
        # Initial upsampling layers
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, 16, 8, padding=4),
            nn.ConvTranspose1d(256, 128, 16, 8, padding=4),
            nn.ConvTranspose1d(128, 64, 4, 2, padding=1),
            nn.ConvTranspose1d(64, 32, 4, 2, padding=1),
        ])
        
        # Residual blocks
        self.resblocks = nn.ModuleList([
            ResBlock(256, kernel_size=3, dilation=(1, 3, 5)),
            ResBlock(128, kernel_size=3, dilation=(1, 3, 5)),
            ResBlock(64, kernel_size=3, dilation=(1, 3, 5)),
            ResBlock(32, kernel_size=3, dilation=(1, 3, 5)),
        ])
        
        # Final conv layer to get audio
        self.final_conv = nn.Conv1d(32, 1, 7, 1, padding=3)
        
    def forward(self, x):
        for i in range(len(self.ups)):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            x = self.resblocks[i](x)
            
        x = F.leaky_relu(x)
        x = self.final_conv(x)
        x = torch.tanh(x)
        
        return x
        
    def inference(self, mel):
        with torch.no_grad():
            return self.forward(mel)
