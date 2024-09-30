import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Solver:
    def __init__(self, args):
        # Existing code...
        self.args = args

        # Load pretrained VGG for perceptual loss
        self.vgg = models.vgg19(pretrained=True).features[:16].eval().to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights

    def perceptual_loss(self, x, y):
        """
        Compute the perceptual loss between x and y using pre-trained VGG19.
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
        # Existing code...
        for i in range(self.args.total_iters):
            # Fetch the next training batch
            x_real, y_real = next(self.train_iter)
            x_real = x_real.to(self.device)
            y_real = y_real.to(self.device)

            # Generate fake images
            z_trg = torch.randn(x_real.size(0), self.args.latent_dim).to(self.device)
            s_trg = self.mapping_network(z_trg, y_real)
            x_fake = self.generator(x_real, s_trg)

            # Compute perceptual loss
            loss_perc = self.perceptual_loss(x_fake, x_real)
            loss_G = args.lambda_perc * loss_perc
            
            # Combine with other generator losses
            # ... existing loss calculations here
            loss_G += existing_losses

            # Backprop and optimization
            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()

            # Logging, saving, and other procedures
            # Existing code continues...
