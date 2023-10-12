import scipy.io as scio
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self, input_dim=361, output_dim=2, hidden_dim=2048):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.BatchNorm1d(num_features=input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Linear(hidden_dim, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 16),
            nn.Linear(16, output_dim),
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        y = self.layers(x)
        y = self.softmax(y)
        return y

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return len(self.data_tensor)

lr = 0.001
epochs = 1000
test_epoch_each = 5

dataset_path = "train(Task_2)"
train_data = torch.tensor(scio.loadmat(os.path.join(dataset_path, "train_data.mat"))["train_data"]).cuda().float()
train_label = torch.tensor(scio.loadmat(os.path.join(dataset_path, "train_label.mat"))["train_label"]).cuda()
train_label = torch.where(train_label == -1, 0, 1).float()
train_label = torch.cat((train_label, torch.cat((torch.zeros(400, 1), torch.ones(400, 1)), dim=0).cuda()), dim=1)

net = MLP().cuda()
#net.load_state_dict(torch.load('mlp.pth'), strict=True)

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,3000,6000,8000], gamma=0.1)
cls_loss = nn.CrossEntropyLoss()

for epoch in range(epochs):
    net.train()

    preds = net(train_data)
    loss = cls_loss(preds, train_label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(epoch, loss, (torch.argmax(preds, dim=1) == torch.argmax(train_label, dim=1)).sum() / 800)

if loss < 0.315:
    torch.save(net.state_dict(), 'mlp.pth')
    print("save model")

