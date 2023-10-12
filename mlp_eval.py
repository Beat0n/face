import torch
import scipy.io as scio
import os
import torch.nn as nn
import numpy

class MLP(nn.Module):
    def __init__(self, input_dim=361, output_dim=2, hidden_dim=2048):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Linear(1024, hidden_dim),
            #nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 1024),
            nn.Linear(1024, 256),
            #nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.Linear(64, 16),
            nn.Linear(16, output_dim),
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        y = self.layers(x)
        y = self.softmax(y)
        return y

def store_txt(label, result_txt):
    with open(result_txt, 'w') as w:
        for index, result in enumerate(label, 1):
            w.write(str(index) + " " + str(result) + "\n")

model = MLP().cuda()
model.load_state_dict(torch.load('mlp.pth'))
test_data = torch.tensor(scio.loadmat("test_data.mat")["test_data"]).cuda().float()
model.eval()
test_preds = torch.argmax(model(test_data), dim=1)
print(test_preds)
test_preds = torch.where(test_preds == 1, -1, 1)
print(test_preds)

store_txt(test_preds.detach().cpu().numpy(), 'mlp_result')