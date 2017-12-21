from collections import OrderedDict

import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from torch.optim import lr_scheduler

from utils.utils import layer_wrapper
from functools import partial, wraps


def normal_weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


class GANLoss(nn.Module):
    def __init__(self, use_gpu):
        super(GANLoss, self).__init__()
        self.use_gpu = use_gpu

    def __call__(self, input, target_value):
        loss_function = nn.BCELoss()
        target_tensor = torch.FloatTensor(input.size()).fill_(target_value)
        if self.use_gpu:
            return loss_function(input, Variable(target_tensor, requires_grad=False).cuda())
        else:
            return loss_function(input, Variable(target_tensor, requires_grad=False))


class VariationalEncoder(nn.Module):
    def __init__(self, k=64, gpu_ids=list()):
        super(VariationalEncoder, self).__init__()
        self.k = k
        self.gpu_ids = gpu_ids

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
        if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
            def model_wrapper(x):
                model = torch.squeeze(self.model(x))
                u = model[..., :self.k]
                sigma = model[..., self.k:]
                return u + Variable(torch.randn([batch_size, self.k]).cuda()) * sigma
            return nn.parallel.data_parallel(model_wrapper, x, self.gpu_ids)
        else:
            model = torch.squeeze(self.model(x))
            u = model[..., :self.k]
            sigma = model[..., self.k:]
            return u + Variable(torch.randn([batch_size, self.k])) * sigma


class VariationalDecoder(nn.Module):
    def __init__(self, k=64, gpu_ids=list()):
        self.k = k
        self.gpu_ids = gpu_ids
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
        if self.gpu_ids and isinstance(z.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, z, self.gpu_ids)
        else:
            return self.model(z)


class Maxout(nn.Module):
    def __init__(self, k=2, gpu_ids=list()):
        self.k = k
        self.gpu_ids = gpu_ids
        super(Maxout, self).__init__()

    def forward(self, x):
        shape = x.size()
        x = x.view((shape[0], -1))
        if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
            def model(x):
                linear = nn.Linear(
                    in_features=x.size()[1],
                    out_features=x.size()[1] * self.k,
                ).cuda()
                output = linear(x)
                output = output.view((shape[0], x.size()[1], 2))
                maxout = torch.max(output, 2)[0]
                return maxout.view(shape)
            return nn.parallel.data_parallel(model, x, self.gpu_ids)
        else:
            linear = nn.Linear(
                in_features=x.size()[1],
                out_features=x.size()[1] * self.k,
            )
            output = linear(x)
            output = output.view((shape[0], x.size()[1], 2))
            maxout = torch.max(output, 2)[0]
            return maxout.view(shape)


class Discriminator(nn.Module):
    def __init__(self, k=64, gpu_ids=list()):
        super(Discriminator, self).__init__()
        self.k = k
        self.gpu_ids = gpu_ids
        self.layer_wrapper = partial(
            layer_wrapper,
            norm_layer=None,
            # activation_function=Maxout(gpu_ids=gpu_ids)
        )

        # Input Format : NCHW
        # Input : 3 x 32 x 32
        self.d_x = nn.Sequential(*[
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
                dropout_rate=0.2,
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=(4, 4),
                    stride=(1, 1)
                ),
                dropout_rate=0.2,
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=(4, 4),
                    stride=(2, 2)
                ),
                dropout_rate=0.2,
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=(4, 4),
                    stride=(1, 1)
                ),
                dropout_rate=0.2,
            ),
        ])

        # Input :  x 1 x 1
        self.d_z = nn.Sequential(*[
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
                dropout_rate=0.2,
            ),
        ])

        self.d_xz = nn.Sequential(*[
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                ),
                dropout_rate=0.2,
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                ),
                dropout_rate=0.2,
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                ),
                dropout_rate=0.2,
                activation_function=nn.Sigmoid()
            )
        ])

    def forward(self, x, z):
        if len(self.gpu_ids) > 0 and isinstance(z.data, torch.cuda.FloatTensor) and\
                isinstance(x.data, torch.cuda.FloatTensor):
            def model(x, z):
                d_x = self.d_x(x)
                z = z.view(z.size()[0], z.size()[1], 1, 1)
                d_z = self.d_z(z)
                d_xz_input = torch.cat((d_x, d_z), 1)
                return torch.squeeze(self.d_xz(d_xz_input))
            return nn.parallel.data_parallel(model, (x, z), self.gpu_ids)
        d_x = self.d_x(x)
        z = z.view(z.size()[0], z.size()[1], 1, 1)
        d_z = self.d_z(z)
        d_xz_input = torch.cat((d_x, d_z), 1)
        return torch.squeeze(self.d_xz(d_xz_input))
