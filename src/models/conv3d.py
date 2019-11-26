"""
TODO create model with 3d conv layers
"""

learning_rate = 1e-2

factor = 8
in_channel = 1
channel_1 = 4 * factor
channel_2 = 8 * factor
channel_3 = 8 * factor
channel_4 = 16 * factor

# torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

model = nn.Sequential(
    #     nn.Conv3d(in_channel, channel_1, kernel_size=7, stride=2, padding=3),
    nn.Conv3d(in_channel, channel_1, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.Conv3d(channel_1, channel_2, kernel_size=5, stride=1, padding=2),
    nn.ReLU(),
    nn.MaxPool3d(kernel_size=2),
    nn.Dropout(p=0.4),

    nn.Conv3d(channel_2, channel_3, kernel_size=5, stride=2, padding=2),
    nn.ReLU(),
    nn.Conv2d(channel_3, channel_4, kernel_size=3, stride=1, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.4),

    Flatten(),
    nn.Linear(1152, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes),
)

optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=0.9,
    nesterov=True,
)

train_part34(model, optimizer, epochs=1)



model = resnet10()

optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=0.9,
    nesterov=True,
)

train_part34(model, optimizer, epochs=1)


def three_layer_convnet(x, params):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?

    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    ################################################################################
    # TODO: Implement the forward pass for the three-layer ConvNet.                #
    ################################################################################

    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params

    # first convolutional layer
    x = F.conv2d(x, conv_w1, padding=2, bias=conv_b1)
    x = F.relu(x)

    # second convolutional layer
    x = F.conv2d(x, conv_w2, padding=1, bias=conv_b2)
    x = F.relu(x)

    # fully connected final layer
    x = flatten(x)  # shape: [batch_size, C x H x W]
    x = x.mm(fc_w) + fc_b

    scores = x

    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################
    return scores


def three_layer_convnet_test():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]

    conv_w1 = torch.zeros((6, 3, 5, 5), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b1 = torch.zeros((6,))  # out_channel
    conv_w2 = torch.zeros((9, 6, 3, 3), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b2 = torch.zeros((9,))  # out_channel

    # you must calculate the shape of the tensor after two conv layers, before the fully-connected layer
    fc_w = torch.zeros((9 * 32 * 32, 10))
    fc_b = torch.zeros(10)

    scores = three_layer_convnet(x, [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b])
    print(scores.size())  # you should see [64, 10]
three_layer_convnet_test()



learning_rate = 3e-3

channel_1 = 32
channel_2 = 16

conv_w1 = None
conv_b1 = None
conv_w2 = None
conv_b2 = None
fc_w = None
fc_b = None

################################################################################
# TODO: Initialize the parameters of a three-layer ConvNet.                    #
################################################################################

C = 10

conv_w1 = random_weight((channel_1, 3, 5, 5))  # Output Channels (Filters), Input Channels, Kernel Height x Width
conv_b1 = torch.zeros((channel_1,), requires_grad=True)  # Output Channels

conv_w2 = random_weight((channel_2, channel_1, 3, 3))
conv_b2 = torch.zeros((channel_2,), requires_grad=True)

fc_w = random_weight((channel_2 * 32 * 32, C))
fc_b = torch.zeros(C, requires_grad=True)

################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
train_part2(three_layer_convnet, params, learning_rate)





