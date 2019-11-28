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

        from src.data.ml import Dataset

        dataset = Dataset()

        # for lr in [1e-1, 1e-2, 1e-3, 1e-4]:
        #     print("\n\nLR: {}".format(lr))
        lr = 1e-2

        fc_model = fc.model(dataset.dimensions)
        # fc_model = fc.Net()
        fc_optimizer = fc.optimizer(fc_model, learning_rate=lr)

        train(
            fc_model,
            fc_optimizer,
            dataset,
            epochs=iargs.epochs,
        )


if __name__ == "__main__":
    run()
