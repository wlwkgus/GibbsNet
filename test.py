from data.data_loader import get_data_loader
from models.models import create_model
from option_parser import TestingOptionParser
from scipy import misc
import numpy as np
import torch
import os

parser = TestingOptionParser()
opt = parser.parse_args()
opt.batch_size = opt.repeat_generation
opt.gpu_ids = []

data_loader = get_data_loader(opt)

model = create_model(opt)
total_steps = 0
model.load(opt.epoch)
Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.FloatTensor
single_input = Tensor(
    1,
    opt.input_channel,
    opt.height,
    opt.width
)
repeated_input = Tensor(
    opt.batch_size,
    opt.input_channel,
    opt.height,
    opt.width
)

for i, data in enumerate(data_loader):
    test_dir = os.path.join(opt.test_dir, opt.model)
    if i >= opt.test_count:
        break
    misc.imsave(test_dir + '/' + 'real_{}.png'.format(i), np.transpose(data[0][0].numpy(), [1, 2, 0]))
    single_input.copy_(
        data[0][0].view(
            1,
            opt.input_channel,
            opt.height,
            opt.width
        )
    )
    repeated_input.copy_(
        single_input.repeat(opt.batch_size, 1, 1, 1)
    )

    model.set_input(repeated_input)
    model.test()

    visuals = model.get_visuals(sample_single_image=False)
    for j in range(opt.batch_size):
        np_image = visuals['fake_x'][j]
        misc.imsave(test_dir + '/' + 'fake_{}_{}.png'.format(i, j), np_image)
