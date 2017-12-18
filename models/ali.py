import torch
from torch.autograd import Variable

from models.base_model import BaseModel
from models.networks import VariationalDecoder, VariationalEncoder, Discriminator


class ALI(BaseModel):
    def __init__(self, opt):
        super(ALI, self).__init__(opt)

        # define input tensors
        self.batch_size = opt.batch_size

        self.encoder = VariationalEncoder()
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=1e-4,
            betas=(0.5, 1e-3)
        )
        self.decoder = VariationalDecoder()
        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=1e-4,
            betas=(0.5, 1e-3)
        )
        self.discriminator = Discriminator()
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=1e-4,
            betas=(0.5, 1e-3)
        )

    def forward(self):
        pass

    def optimize_parameters(self):
        pass

    def backward(self):
        pass
