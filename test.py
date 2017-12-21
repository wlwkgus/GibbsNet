from data.data_loader import get_data_loader
from models.models import create_model
from option_parser import TestingOptionParser
from scipy import misc
import numpy as np

parser = TestingOptionParser()
opt = parser.parse_args()
opt.batch_size = 1

data_loader = get_data_loader(opt)

model = create_model(opt)
total_steps = 0
model.load()


for i, data in enumerate(data_loader):
    if i >= opt.test_count:
        break
    misc.imsave('real_{}.png'.format(i), np.transpose(data[0][0].numpy(), [1, 2, 0]))
    repeated_input = data[0].repeat([opt.repeat_generation, 1, 1, 1])
    model.set_input(repeated_input)
    model.test()
    visuals = model.get_visuals()
    for j in range(opt.repeat_generation):
        image_tensor = visuals['fake_x'][j]
        misc.imsave('fake_{}_{}.png'.format(i, j), np.transpose(image_tensor.numpy(), [1, 2, 0]))
