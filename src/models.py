import torch
from torch import nn



def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

class ConvNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_blocks=3, epochs=5, lr=1e-3, class_weights=None, weight_decay=1e-2):
        super().__init__()
        layers = [conv_block(input_dim, hidden_dim)]
        for _ in range(num_blocks - 1):
            layers.append(conv_block(hidden_dim, hidden_dim))
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, 1)

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weights = class_weights if class_weights is not None else [0.4, 0.6]

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2,3))  # GAP
        return self.classifier(x).flatten()

    def configure_optimizer(self, name="adam", lr=None, weight_decay=None):
        if lr is None:
            lr = self.lr
        if weight_decay is None:
            weight_decay = self.weight_decay
        name = name.lower()
        params = self.parameters()
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    @torch.no_grad()
    def predict_proba(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        return torch.sigmoid(self.forward(x))