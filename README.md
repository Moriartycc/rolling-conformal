# rolling-conformal

Code for the numerical experiments in the paper **Rolling Conformal Prediction
in Sequential Training**.

Authors: Chen Cheng, Ruiting Liang, and Rina Foygel Barber.

The repository contains the three experiments reported in the paper:

1. `linear-regression`: minimum-norm OLS along a growing data stream.
2. `sgd-logistic`: online SGD for five-class logistic regression.
3. `one-pass-mnist`: one-pass training of a LeNet-style network on MNIST.

Each folder is self-contained and has its own requirements and documentation.
The default command reproduces the corresponding paper configuration and
writes only PNG and PDF figures directly to that experiment's `output` folder.
The included figures were produced by the full paper runs.

## Quick start

Enter an experiment folder, install its dependencies, and run:

```bash
python -m pip install -r requirements.txt
bash run_experiment.sh
```

The defaults are the full-scale paper settings and can be computationally
expensive. Each experiment README gives a smaller command for testing the
installation.

## Figure map

| Experiment | Paper figures |
| --- | --- |
| Linear regression | `fig1-1` through `fig1-3` |
| Logistic SGD | `fig2-1` through `fig2-6` |
| One-pass MNIST | `fig3-1` and `fig3-2` |

