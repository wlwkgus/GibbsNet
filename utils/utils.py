from torch import nn


def layer_wrapper(
        layer,
        norm_layer=nn.BatchNorm2d,
        dropout_rate=0.0,
        activation_function=nn.LeakyReLU(negative_slope=0.1)
):
    layers = list()
    layers.append(layer)
    if norm_layer:
        if type(layers[-1]) == nn.Conv2d:
            layers.append(norm_layer(layers[-1].out_channels))
        elif type(layers[-1]) == nn.ConvTranspose2d:
            layers.append(norm_layer(layers[-1].out_channels))
        else:
            layers.append(norm_layer(layers[-1].out_channels))
    if dropout_rate > 0.0:
        layers.append(nn.Dropout(dropout_rate))
    if activation_function:
        layers.append(activation_function)

    return nn.Sequential(*layers)
