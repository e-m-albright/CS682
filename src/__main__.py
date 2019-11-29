"""
Project entry point, primarily pick your desired model framework and launch training/evaluation
"""
import os
import pickle


def run():
    from src.args import iargs

    if iargs.model == "svm":
        from src.models.sklearn import svm
        print(svm.test_svm())

    elif iargs.model == "mlp":
        from src.models.sklearn import mlp
        print(mlp.test_mlp())

    elif iargs.model == "fc":
        from src.data.ml import Dataset
        from src.models import fc
        from src.utils import plotting
        from src.utils.training import train

        dataset = Dataset(dimensions='3d')

        model = fc.Net(dataset.dimensions)
        optimizer = fc.optimizer(model, learning_rate=1e-1)
        criterion = fc.criterion()

        losses, accuracies = train(
            model,
            optimizer,
            criterion,
            dataset,
            epochs=iargs.epochs,
            print_frequency=iargs.print_freq,
        )

        if iargs.plot:
            plotting.plot_loss(losses)
            plotting.plot_accuracies(accuracies)

    elif iargs.model in ["conv2d", "2d"]:
        from src.data.ml import Dataset
        from src.models import conv2d
        from src.utils import plotting
        from src.utils.training import train

        dataset = Dataset(dimensions='2d')

        model = conv2d.model()
        optimizer = conv2d.optimizer(model, learning_rate=1e-2)
        criterion = conv2d.criterion()

        losses, accuracies = train(
            model,
            optimizer,
            criterion,
            dataset,
            epochs=iargs.epochs,
            print_frequency=iargs.print_freq,
        )

        if iargs.plot:
            plotting.plot_loss(losses)
            plotting.plot_accuracies(accuracies)

    elif iargs.model in ["conv3d", "3d"]:
        from src.data.ml import Dataset
        from src.models import conv3d, resnet3d
        from src.utils.training import train

        dset_path = "saved_dataset.pkl"
        # if os.path.exists(dset_path):
        #     with open(dset_path, 'rb') as f:
        #         dataset = pickle.load(f)
        # else:
        dataset = Dataset(dimensions='3d', scale=1./2, limit=10)
        with open(dset_path, 'wb') as f:
            pickle.dump(dataset, f)

        print(dataset.dimensions)

        # model = conv3d.Net(dataset.dimensions)
        model = resnet3d.r3d_18()
        optimizer = conv3d.optimizer(model, learning_rate=1e-2)
        criterion = conv3d.criterion()

        train(
            model,
            optimizer,
            criterion,
            dataset,
            epochs=iargs.epochs,
            print_frequency=iargs.print_freq,
        )


if __name__ == "__main__":
    run()
