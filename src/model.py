import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    """Spatial Transformer Network front-end.

    Learns an affine warp that rectifies each input image (slant, scale,
    baseline skew) before the CRNN reads it. Milestone 2: directly targets
    the geometric variability of children's handwriting called out in the
    project plan. Initialised to the identity so training starts from the
    baseline behaviour and only learns corrections it needs.
    """

    def __init__(self):
        super().__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(1, 16, 7, padding=3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )
        # Start as the identity transform.
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        xs = self.loc(x).flatten(1)
        theta = self.fc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


class CRNN(nn.Module):
    def __init__(self, num_classes, rnn_hidden=256, rnn_layers=2, dropout=0.3,
                 use_stn=False):
        super().__init__()

        self.stn = STN() if use_stn else None

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(512, 512, (2, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0,
            batch_first=False,
        )

        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        if self.stn is not None:
            x = self.stn(x)
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, f"CNN output height must be 1, got {h}"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)

        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        log_probs = F.log_softmax(output, dim=2)
        return log_probs
