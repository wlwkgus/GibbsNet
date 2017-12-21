from models.ali import ALI
from models.gibbs_net import GibbsNet


def create_model(opt):
    if opt.model == 'ALI':
        return ALI(opt)
    elif opt.model == 'GibbsNet':
        return GibbsNet(opt)
    else:
        raise Exception("This implementation only supports ALI, GibbsNet.")
