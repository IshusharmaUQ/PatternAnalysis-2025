import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=10)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

def contrastive_loss(output1, output2, target, margin=1.0):
    euclidean_distance = F.pairwise_distance(output1, output2)
    loss = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                      (target) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss
