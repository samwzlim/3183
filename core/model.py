"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import math
import torchvision
from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.attention import SpatialTransformer, BasicTransformerBlock
from core.wing import FAN

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, args, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.args = args
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        if self.args.use_cross_attention:
            self.cross_att1 = SpatialTransformer(dim_in, 8, 64, context_dim=style_dim)
            self.cross_att2 = SpatialTransformer(dim_out, 8, 64, context_dim=style_dim)
        else:
            self.norm1 = AdaIN(style_dim, dim_in)
            self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        if self.args.use_cross_attention:
            x = self.cross_att1(x, s.unsqueeze(1))
        else:
            x = self.norm1(x,s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        if self.args.use_cross_attention:
            x = self.cross_att2(x, s.unsqueeze(1))
        else:
            x = self.norm2(x,s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, args, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(args, dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(args, dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])
        return self.to_rgb(x)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s

class StyleEncoderSEAN(nn.Module):
    def __init__(self, args, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        self.args = args
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
     
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 3, 1, 1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]
        if self.args.use_self_attention:
            self_attention_layer = 4
            self.att = nn.ModuleList()
            for _ in range(self_attention_layer):
                self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(style_dim, 8, 64)
                for d in range(self_attention_layer)]
        )

        

    def forward(self, x, y, mask):
        h = self.shared(x)
        resize = torchvision.transforms.Resize((64,64),antialias=True)
        mask = resize(mask)
         
        # SEAN encoder
        b_size = h.shape[0]
        s_size = mask.shape[1]
        f_size = h.shape[1]
        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=h.dtype, device=h.device)


        for i in range(b_size):
            for j in range(s_size):
                component_mask_area = torch.sum(mask.bool()[i, j])
                if component_mask_area > 0:
                    codes_component_feature = h[i].masked_select(mask.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature

                    # codes_avg[i].masked_scatter_(segmap.bool()[i, j], codes_component_mu)
        h = codes_vector.squeeze(0)
        h = h.view(h.size(0), -1) 
        # stargan-v2 encoder linear
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        if self.args.use_self_attention:
            for layer in self.att:
                s = layer(s)
        return s
    
class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


def build_model(args):
    generator = nn.DataParallel(Generator(args, args.img_size, args.style_dim, w_hpf=args.w_hpf), device_ids=[args.gpu_id])
    mapping_network = nn.DataParallel(MappingNetwork(args.latent_dim, args.style_dim, args.num_domains), device_ids=[args.gpu_id])
    if args.use_sean_encoder:   
        style_encoder = nn.DataParallel(StyleEncoderSEAN(args, args.img_size, args.style_dim, args.num_domains), device_ids=[args.gpu_id])
    else:
        style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains), device_ids=[args.gpu_id])
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains), device_ids=[args.gpu_id])
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    torch.set_float32_matmul_precision('high')
    
    if args.use_torch_compile:
        generator = torch.compile(generator, dynamic=False)
        generator_ema = torch.compile(generator_ema, dynamic=False)
        mapping_network = torch.compile(mapping_network, dynamic=False)
        mapping_network_ema = torch.compile(mapping_network_ema, dynamic=False)
        discriminator2 = torch.compile(discriminator, dynamic=False)
        style_encoder = torch.compile(style_encoder, dynamic=False)
        style_encoder_ema = torch.compile(style_encoder_ema, dynamic=False)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    if args.w_hpf > 0:
        fan = nn.DataParallel(FAN(fname_pretrained=args.wing_path).eval())
        fan.get_heatmap = fan.module.get_heatmap
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema