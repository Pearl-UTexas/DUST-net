import torch
import torch.nn.functional as F
from torchvision import models


class VMSoftOrthoNet(torch.nn.Module):
    def __init__(self, hidden_size=[], img_seq_len=16):
        super(VMSoftOrthoNet, self).__init__()
        self.convnet = models.resnet18(pretrained=True)
        layers = []
        self.resnet_out_classes = 1000
        self.img_seq_len = img_seq_len
        n_in = self.img_seq_len * self.resnet_out_classes  # resnet output size
        for n_out in hidden_size:
            layers.append(torch.nn.Linear(n_in, n_out))
            layers.append(torch.nn.ReLU())
            n_in = n_out

        # Add output layer
        layers.append(torch.nn.Linear(n_in, 40))
        # layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Softplus())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, X):
        cnn_embed_seq = []
        for t in range(X.size(1)):
            x = self.convnet(X[:, t, :, :, :])
            x = x.view(x.size(0), -1)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        x = cnn_embed_seq.contiguous().view(
            -1, self.img_seq_len * self.resnet_out_classes
        )
        x = self.mlp.forward(x)
        return x


class VMStiefelNet(torch.nn.Module):
    def __init__(self, hidden_size=[], img_seq_len=16):
        super().__init__()
        self.convnet = models.resnet18(pretrained=True)
        layers = []
        self.resnet_out_classes = 1000
        self.img_seq_len = img_seq_len
        n_in = self.img_seq_len * self.resnet_out_classes  # resnet output size
        for n_out in hidden_size:
            layers.append(torch.nn.Linear(n_in, n_out))
            layers.append(torch.nn.ReLU())
            n_in = n_out

        # Add output layer
        layers.append(torch.nn.Linear(n_in, 40))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, X):
        cnn_embed_seq = []
        for t in range(X.size(1)):
            x = self.convnet(X[:, t, :, :, :])
            x = x.view(x.size(0), -1)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        x = cnn_embed_seq.contiguous().view(
            -1, self.img_seq_len * self.resnet_out_classes
        )
        x = self.mlp.forward(x)

        # We want to apply softplus on a subset of x (everywhere except F_mat)
        x = torch.cat((x[:, :6], F.softplus(x[:, 6:])), dim=-1)
        return x


class VMStiefelSVDNet(VMStiefelNet):
    def __init__(self, hidden_size=[], img_seq_len=16):
        super(VMStiefelSVDNet, self).__init__(hidden_size, img_seq_len)

    def forward(self, X):
        cnn_embed_seq = []
        for t in range(X.size(1)):
            x = self.convnet(X[:, t, :, :, :])
            x = x.view(x.size(0), -1)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        x = cnn_embed_seq.contiguous().view(
            -1, self.img_seq_len * self.resnet_out_classes
        )
        x = self.mlp.forward(x)

        # We want to apply softplus on a subset of x (everywhere except F_mat)
        x = torch.cat(
            (
                F.relu6(x[:, :4]),
                #torch.sigmoid(x[:, :4]),
                F.softplus(x[:, 4:]),
            ),
            dim=-1,
        )
        return x
