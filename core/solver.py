"""
StarGAN v2 Solver with Perceptual Loss Integration
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from munch import Munch
from torchvision import models

from core.wing import FAN


class Solver:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()
        self.optimizer_G = torch.optim.Adam(self.nets.generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.optimizer_D = torch.optim.Adam(self.nets.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

        # Load pre-trained VGG19 for perceptual loss
        self.vgg = models.vgg19(pretrained=True).features[:16].eval().to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights

    def build_model(self):
        """Build Generator, Discriminator, and other components"""
        from core.model import Generator, MappingNetwork, StyleEncoder, Discriminator, FAN
        self.nets = Munch(generator=Generator(args.img_size, args.style_dim, args.w_hpf),
                          mapping_network=MappingNetwork(args.latent_dim, args.style_dim, args.num_domains),
                          style_encoder=StyleEncoder(args.img_size, args.style_dim, args.num_domains),
                          discriminator=Discriminator(args.img_size, args.num_domains))
        self.nets.generator.to(self.device)
        self.nets.mapping_network.to(self.device)
        self.nets.style_encoder.to(self.device)
        self.nets.discriminator.to(self.device)

        if args.w_hpf > 0:
            self.fan = FAN(fname_pretrained=args.wing_path).eval().to(self.device)
            self.nets.fan = self.fan

    def perceptual_loss(self, x, y):
        """
        Compute perceptual loss between x and y using VGG19.
        Args:
            x: Generated image
            y: Real image
        Returns:
            Perceptual loss (L1 loss between feature maps)
        """
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return F.l1_loss(x_features, y_features)

    def train(self, loaders):
        """Training loop."""
        self.train_iter = iter(loaders.src)
        self.val_iter = iter(loaders.val)

        for i in range(self.args.total_iters):
            # Fetch training batch
            try:
                x_real, y_org = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(loaders.src)
                x_real, y_org = next(self.train_iter)

            x_real = x_real.to(self.device)
            y_org = y_org.to(self.device)

            # Train Discriminator
            d_loss = self.compute_d_loss(x_real, y_org)
            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()

            # Train Generator
            g_loss = self.compute_g_loss(x_real, y_org)
            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()

            # Logging
            if (i + 1) % self.args.print_every == 0:
                print(f"Iter [{i+1}/{self.args.total_iters}] | d_loss: {d_loss.item():.4f} | g_loss: {g_loss.item():.4f}")

            # Save the model checkpoints
            if (i + 1) % self.args.save_every == 0:
                torch.save(self.nets.generator.state_dict(), f'expr/checkpoints/generator_{i+1}.pth')
                torch.save(self.nets.discriminator.state_dict(), f'expr/checkpoints/discriminator_{i+1}.pth')

    def compute_d_loss(self, x_real, y_org):
        """Compute loss for the discriminator."""
        x_fake = self.nets.generator(x_real, y_org)
        out_real = self.nets.discriminator(x_real, y_org)
        out_fake = self.nets.discriminator(x_fake.detach(), y_org)

        # Adversarial loss
        d_loss_real = torch.mean(F.relu(1.0 - out_real))
        d_loss_fake = torch.mean(F.relu(1.0 + out_fake))

        return d_loss_real + d_loss_fake

    def compute_g_loss(self, x_real, y_org):
        """Compute loss for the generator."""
        # Adversarial loss
        x_fake = self.nets.generator(x_real, y_org)
        out_fake = self.nets.discriminator(x_fake, y_org)
        g_loss_adv = -torch.mean(out_fake)

        # Perceptual Loss
        g_loss_perc = self.perceptual_loss(x_fake, x_real) * self.args.lambda_perc

        return g_loss_adv + g_loss_perc

    @torch.no_grad()
    def sample(self, loaders):
        """Sample images for testing."""
        self.nets.generator.eval()
        for x_src, y_src in loaders.src:
            x_src = x_src.to(self.device)
            y_src = y_src.to(self.device)

            x_fake = self.nets.generator(x_src, y_src)
            x_fake = (x_fake + 1) / 2  # Convert back to [0, 1]
            torchvision.utils.save_image(x_fake, f'expr/samples/{i+1}_fake.png')