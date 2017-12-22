from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch import nn

from models.base_model import BaseModel
from models.networks import VariationalDecoder, VariationalEncoder, Discriminator, GANLoss, normal_weight_init
from utils.utils import tensor2im


class ALI(BaseModel):
    def __init__(self, opt):
        super(ALI, self).__init__(opt)

        # define input tensors
        self.gpu_ids = opt.gpu_ids
        self.batch_size = opt.batch_size

        self.encoder = VariationalEncoder(gpu_ids=self.gpu_ids, k=self.opt.z_dimension)
        self.decoder = VariationalDecoder(gpu_ids=self.gpu_ids, k=self.opt.z_dimension)
        if self.gpu_ids:
            self.encoder.cuda(device=opt.gpu_ids[0])
            self.decoder.cuda(device=opt.gpu_ids[0])
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.opt.lr,
            betas=(0.5, 1e-3)
        )
        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=self.opt.lr,
            betas=(0.5, 1e-3)
        )
        self.discriminator = Discriminator(gpu_ids=self.gpu_ids)
        if self.gpu_ids:
            self.discriminator.cuda(device=opt.gpu_ids[0])
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.opt.lr,
            betas=(0.5, 1e-3)
        )

        # normal initialization.
        self.encoder.apply(normal_weight_init)
        self.decoder.apply(normal_weight_init)
        self.discriminator.apply(normal_weight_init)

        assert self.decoder.k == self.encoder.k

        # input
        self.input = self.Tensor(
            opt.batch_size,
            opt.input_channel,
            opt.height,
            opt.width
        )
        self.x = None

        self.normal_z = None
        self.sampled_x = None
        self.sampled_z = None
        self.d_sampled_x = None
        self.d_sampled_z = None

        # losses
        self.loss_function = GANLoss(len(self.gpu_ids) > 0)
        self.D_loss = None
        self.G_loss = None

    def set_input(self, data, is_z_given=False):
        temp = self.input.clone()
        temp.resize_(self.input.size())
        temp.copy_(self.input)
        self.input = temp
        self.input.resize_(data.size()).copy_(data)
        if not is_z_given:
            self.set_z()

    def set_z(self, var=None, volatile=False):
        if var is None:
            self.normal_z = var
        else:
            if self.gpu_ids:
                self.normal_z = Variable(torch.randn((self.opt.batch_size, self.encoder.k)).cuda(), volatile=volatile)
            else:
                self.normal_z = Variable(torch.randn((self.opt.batch_size, self.encoder.k)), volatile=volatile)

    def forward(self, volatile=False):
        # volatile : no back gradient.
        self.x = Variable(self.input, volatile=volatile)
        # Before call self.decoder, normal_z must be set.
        self.sampled_x = self.decoder(self.normal_z)
        self.sampled_z = self.encoder(self.x)

        if not volatile:
            self.d_sampled_x = self.discriminator(self.x, self.sampled_z)
            self.d_sampled_z = self.discriminator(self.sampled_x, self.normal_z)

    def test(self):
        self.forward(volatile=True)

    def forward_encoder(self, var):
        return self.encoder(var)

    def forward_decoder(self, var):
        return self.decoder(var)

    def optimize_parameters(self):
        self.forward()

        # update discriminator
        self.discriminator_optimizer.zero_grad()
        self.backward_D()
        self.discriminator_optimizer.step()
        # update generator
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.backward_G()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def backward_D(self):
        self.D_loss = self.loss_function(
            self.d_sampled_x, 1.
        ) + self.loss_function(
            self.d_sampled_z, 0.
        )
        self.D_loss.backward(retain_graph=True)

    def backward_G(self):
        self.G_loss = self.loss_function(
            self.d_sampled_x, 0.
        ) + self.loss_function(
            self.d_sampled_z, 1.
        )
        self.G_loss.backward(retain_graph=True)

    def get_losses(self):
        return OrderedDict([
            ('D_loss', self.D_loss.cpu().data.numpy()[0]),
            ('G_loss', self.G_loss.cpu().data.numpy()[0]),
        ])

    def get_visuals(self, sample_single_image=True):
        fake_x = tensor2im(self.sampled_x.data, sample_single_image=sample_single_image)
        real_x = tensor2im(self.x.data, sample_single_image=sample_single_image)
        return OrderedDict([('real_x', real_x), ('fake_x', fake_x)])

    def save(self, epoch):
        self.save_network(self.encoder, 'encoder', epoch, self.gpu_ids)
        self.save_network(self.decoder, 'decoder', epoch, self.gpu_ids)
        self.save_network(self.discriminator, 'discriminator', epoch, self.gpu_ids)

    def load(self, epoch):
        self.load_network(self.encoder, 'encoder', epoch)
        self.load_network(self.decoder, 'decoder', epoch)
        self.load_network(self.discriminator, 'discriminator', epoch)

    def remove(self, epoch):
        if epoch == 0:
            return
        self.remove_checkpoint('encoder', epoch)
        self.remove_checkpoint('decoder', epoch)
        self.remove_checkpoint('discriminator', epoch)
