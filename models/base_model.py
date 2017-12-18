import torch
import os


class BaseModel(object):
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.FloatTensor
        self.save_dir = os.path.join(opt.ckpt_dir, opt.name)

    @property
    def name(self):
        return 'BaseModel'

    def save_network(self, network, network_name, epoch_count, gpu_ids):
        save_filename = ""
        save_path = ""
        # TODO : write latest epoch per network_name
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device_id=gpu_ids[0])

    def load_network(self, network, network_name, epoch_count):
        # TODO : write latest epoch per network_name
        save_filename = ""
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_checkpoint(self):
        pass

