"""
TODO

Awesome!

Some TODO's to work on in the future
- More subjects (memory management)
- Cut timesteps that could either belong to one event or another (boundary dispute) as I'm not 100% confident in event assignment

- Deep learn, VGG
- Find convolution on non flattened image (3d convolution could get brain regions better?)

- Review HW for good ideas in training, data manipulation, etc

"""


def run():
    from src.args import iargs

    if iargs.model == "svm":
        from src.models.sklearn import svm
        print(svm.test_svm())

    elif iargs.model == "mlp":
        from src.models.sklearn import mlp
        print(mlp.test_mlp())

    elif iargs.model == "fc":
        from src.models import fc
        from src.utils.training import train

        # l_train, l_validate, l_test = get_loaders()

        import torch
        import sklearn.datasets
        x, y = sklearn.datasets.make_moons(200, noise=0.2)
        x = torch.from_numpy(x).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)

        from torch.utils.data import DataLoader, TensorDataset

        loader = DataLoader(TensorDataset(x, y))

        # x, y = next(iter(l_train))
        # print(x[0, 8:14, 20:32, 20:32])
        # import numpy as np
        # dimensions = np.prod(X.shape[1:])

        # for lr in [1e-1, 1e-2, 1e-3, 1e-4]:
        #     print("\n\nLR: {}".format(lr))
        lr = 1e-2

        # fc_model = fc.model(dimensions)
        fc_model = fc.Net()
        fc_optimizer = fc.optimizer(fc_model, learning_rate=lr)

        train(
            fc_model,
            fc_optimizer,
            loader,
            loader,
            # l_train,
            # l_validate,
            epochs=iargs.epochs,
        )


if __name__ == "__main__":
    run()
