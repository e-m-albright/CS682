"""
Project entry point, primarily pick your desired model framework and launch training/evaluation
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
        # epochs recommended above 80 - 100
        fc.run(iargs)

    elif iargs.model in ["conv2d", "2d"]:
        from src.models import conv2d
        conv2d.run(iargs)

    elif iargs.model in ["conv3d", "3d"]:
        from src.models import conv3d
        conv3d.run(iargs)


if __name__ == "__main__":
    run()
