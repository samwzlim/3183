import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from skimage.filters import gaussian
from collections import namedtuple
from munch import Munch

# Additional functions and imports for processing landmarks

class FAN(nn.Module):
    def __init__(self, num_modules=1, end_relu=False, num_landmarks=106, fname_pretrained=None):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.end_relu = end_relu

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.hourglass = HourGlass(1, 4, 256)
        self.top_m = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(256, 106 + 1, kernel_size=1, stride=1, padding=0)

        if fname_pretrained:
            self.load_pretrained_weights(fname_pretrained)

    def load_pretrained_weights(self, fname):
        if torch.cuda.is_available():
            checkpoint = torch.load(fname)
        else:
            checkpoint = torch.load(fname, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['state_dict'])

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = F.relu(self.conv3(x), True)
        x = self.conv4(x)

        x = self.hourglass(x)
        x = self.top_m(x)
        x = self.conv_last(x)
        return x


class FaceAligner():
    def __init__(self, fname_wing, fname_celeba_mean, output_size=256):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fan = FAN(fname_pretrained=fname_wing).to(self.device).eval()
        scale = output_size // 256
        self.CELEB_REF = np.float32(np.load(fname_celeba_mean)['mean']) * scale
        self.output_size = output_size

    def align(self, imgs):
        imgs = imgs.to(self.device)
        landmarkss = self.fan(imgs).cpu().numpy()
        for i, landmarks in enumerate(landmarkss):
            img_np = tensor2numpy255(imgs[i])
            transform = self.landmarks2mat(landmarks)
            aligned = cv2.warpPerspective(img_np, transform, (self.output_size, self.output_size), flags=cv2.INTER_LANCZOS4)
            imgs[i] = np2tensor(aligned[:self.output_size, :self.output_size, :])
        return imgs

    def landmarks2mat(self, landmarks):
        T_origin = points2T(landmarks, 'from')
        eye_center, mouth_center = get_eye_mouth_centers(landmarks)
        S = landmarks2S(landmarks, self.CELEB_REF)
        T_ref = points2T(self.CELEB_REF, 'to')
        matrix = np.dot(T_ref, np.dot(S, T_origin))
        return matrix
