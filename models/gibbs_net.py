from models.ali import ALI
from models.base_model import BaseModel
from torch.autograd import Variable
import torch


class GibbsNet(BaseModel):
    def __init__(self, opt):
        super(GibbsNet, self).__init__(opt)

        self.ali_model = ALI(opt)
        self.opt = opt
        self.sampling_count = opt.sampling_count
        self.z = None
        self.fake_x = None

        self.input = self.Tensor(
            opt.batch_size,
            opt.input_channel,
            opt.height,
            opt.width
        )
        self.x = None

    def forward(self, volatile=False):
        self.sampling()

        # clamped chain : ALI model
        self.ali_model.set_z(var=self.z)
        self.ali_model.set_input(self.x.data, is_z_given=True)

        self.ali_model.forward()

    def test(self):
        self.forward(volatile=True)

    def set_input(self, data):
        temp = self.input.clone()
        temp.resize_(self.input.size())
        temp.copy_(self.input)
        self.input = temp
        self.input.resize_(data.size()).copy_(data)

    def sampling(self, volatile=True):
        batch_size = self.opt.batch_size
        self.x = Variable(self.input, volatile=volatile)

        # unclamped chain
        if self.gpu_ids:
            self.z = Variable(torch.randn((batch_size, self.ali_model.encoder.k)).cuda(), volatile=volatile)
        else:
            self.z = Variable(torch.randn((batch_size, self.ali_model.encoder.k)), volatile=volatile)

        for i in range(self.sampling_count):
            self.fake_x = self.ali_model.decoder(self.z)
            self.z = self.ali_model.encoder(self.fake_x)

    def save(self, epoch):
        self.ali_model.save(epoch)

    def load(self, epoch):
        self.ali_model.load(epoch)

    def optimize_parameters(self):
        self.sampling()
        # clamped chain : ALI model
        self.ali_model.set_z(var=self.z)
        self.ali_model.set_input(self.x.data, is_z_given=True)

        self.ali_model.optimize_parameters()

    def get_losses(self):
        return self.ali_model.get_losses()

    def get_visuals(self, sample_single_image=True):
        return self.ali_model.get_visuals(sample_single_image=sample_single_image)

    def remove(self, epoch):
        self.ali_model.remove(epoch)
