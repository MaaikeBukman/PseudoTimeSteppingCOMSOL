import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_in, num_neurons, num_layers, dropout):
        super().__init__()
        layers = [nn.Linear(num_in, num_neurons), nn.LeakyReLU()]
        for _ in range(num_layers-1):
            layers += [nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(num_neurons, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)