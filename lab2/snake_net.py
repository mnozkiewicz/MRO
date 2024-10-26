import torch.nn as nn
import torch

def convolutions(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3)),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),

        nn.Conv2d(out_channels, out_channels, (3, 3)),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )

class Snake_Net(nn.Module):
    
    def __init__(self, contract: int, dropout_rate: float):
        super().__init__()

        self.contract = contract

        self.pooling = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout2d(dropout_rate)

        self.encoders = nn.ModuleList([
            nn.ModuleList([
                convolutions(3, 32),
                convolutions(32, 64),
                convolutions(64, 128)
            ])
            for _ in range(contract)
        ])

        self.decoders = nn.ModuleList([
            nn.ModuleList([
                convolutions((i + 2)*64, 64),
                convolutions((i + 2)*32, 32),
            ])
            for i in range(contract)
        ])

        
        self.expanders = nn.ModuleList([
            nn.ModuleList([
                nn.ConvTranspose2d(128, 64, (10, 10), stride=2),
                nn.ConvTranspose2d(64, 32, (18, 18), stride=2),
                nn.ConvTranspose2d(32, 3, (9, 9), stride=1)
            ])
            for _ in range(contract)
        ])

        self.final = nn.Sequential(
            convolutions(3, 32),
            self.pooling,
            self.dropout,
            convolutions(32, 64),
            self.pooling,
            self.dropout,
            convolutions(64, 128),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )


    def forward(self, x):

        previous_blocks = []

        for i in range(self.contract):

            encoder = self.encoders[i]
            decoder = self.decoders[i]
            expander = self.expanders[i]

            new_blocks = []

            for convolve in encoder[:-1]:
                new_blocks.append(x := convolve(x))
                x = self.pooling(x)
                x = self.dropout(x)

            previous_blocks.append(new_blocks)

            x = encoder[-1](x)

            for j, expand in enumerate(expander[:-1]):
                x = expand(x)
                x = torch.concat([block[-1-j] for block in previous_blocks] + [x], 1)
                x = decoder[j](x)

            x = expander[-1](x)
        
        x = self.final(x)
        return x
    
    