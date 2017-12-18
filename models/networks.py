from collections import OrderedDict

import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from torch.optim import lr_scheduler

from utils.utils import layer_wrapper
from functools import partial


class VariationalEncoder(nn.Module):
    def __init__(self, k=64):
        super(VariationalEncoder, self).__init__()
        self.k = k

        # Format : NCHW
        model = [
            layer_wrapper(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=32,
                    kernel_size=(5, 5),
                    stride=(1, 1)
                )
            ),
            layer_wrapper(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=(4, 4),
                    stride=(2, 2)
                )
            ),
            layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=(4, 4),
                    stride=(1, 1)
                )
            ),
            layer_wrapper(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=(4, 4),
                    stride=(2, 2)
                )
            ),
            layer_wrapper(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=(4, 4),
                    stride=(1, 1)
                )
            ),
            layer_wrapper(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                )
            ),
            layer_wrapper(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=k * 2,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                ),
                norm_layer=None,
                activation_function=None
            ),
        ]
        self.model = nn.Sequential(
            *model
        )

    def forward(self, x):
        batch_size = x.size()[0]
        model = torch.squeeze(self.model(x))
        u = model[..., :self.k]
        sigma = model[..., self.k:]
        return u + Variable(torch.randn([batch_size, self.k])) * sigma


class VariationalDecoder(nn.Module):
    def __init__(self, k=64):
        super(VariationalDecoder, self).__init__()

        # Format : NCHW
        model = [
            layer_wrapper(
                nn.ConvTranspose2d(
                    in_channels=k,
                    out_channels=256,
                    kernel_size=(4, 4),
                    stride=(1, 1)
                )
            ),
            layer_wrapper(
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=(4, 4),
                    stride=(2, 2)
                )
            ),
            layer_wrapper(
                nn.ConvTranspose2d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=(4, 4),
                    stride=(1, 1)
                )
            ),
            layer_wrapper(
                nn.ConvTranspose2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=(4, 4),
                    stride=(2, 2)
                )
            ),
            layer_wrapper(
                nn.ConvTranspose2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=(5, 5),
                    stride=(1, 1)
                )
            ),
            layer_wrapper(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                )
            ),
            layer_wrapper(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=3,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                ),
                norm_layer=None,
                activation_function=nn.Sigmoid()
            ),
        ]
        self.model = nn.Sequential(
            *model
        )

    def forward(self, z):
        z = z.view([z.size()[0], z.size()[1], 1, 1])
        model = self.model(z)
        print(model)
        return model


class Maxout(nn.Module):
    def __init__(self, k=2):
        self.k = k
        super(Maxout, self).__init__()

    def forward(self, x):
        shape = x.size()
        x = x.view((shape[0], -1))
        linear = nn.Linear(
            in_features=x.size()[1],
            out_features=x.size()[1] * self.k,
        )
        output = linear(x)
        output = output.view((shape[0], x.size()[1], 2))
        maxout = torch.max(output, 2)[0]
        return maxout.view(shape)


class Discriminator(nn.Module):
    def __init__(self, k=64):
        super(Discriminator, self).__init__()
        self.k = k
        self.layer_wrapper = partial(
            layer_wrapper,
            norm_layer=None,
            activation_function=Maxout()
        )

        # Input Format : NCHW
        # Input : 3 x 32 x 32
        self.d_x = nn.Sequential([
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=32,
                    kernel_size=(5, 5),
                    stride=(1, 1)
                ),
                dropout_rate=0.2,
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=(4, 4),
                    stride=(2, 2)
                ),
                dropout_rate=0.5,
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=(4, 4),
                    stride=(1, 1)
                ),
                dropout_rate=0.5,
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=(4, 4),
                    stride=(2, 2)
                ),
                dropout_rate=0.5,
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=(4, 4),
                    stride=(1, 1)
                ),
                dropout_rate=0.5,
            ),
        ])

        # Input :  x 1 x 1
        self.d_z = nn.Sequential([
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=self.k,
                    out_channels=512,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                ),
                dropout_rate=0.2,
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                ),
                dropout_rate=0.5,
            ),
        ])

        self.d_xz = nn.Sequential([
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                ),
                dropout_rate=0.5,
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                ),
                dropout_rate=0.5,
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                ),
                dropout_rate=0.5,
                activation_function=nn.Sigmoid()
            )
        ])

    def forward(self, x, z):
        d_x = self.d_x(x)
        z = z.view(z.size()[0], z.size()[1], 1, 1)
        d_z = self.d_z(z)
        d_xz_input = torch.cat((d_x, d_z), 1)
        return torch.squeeze(self.d_xz(d_xz_input))
