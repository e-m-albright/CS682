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


