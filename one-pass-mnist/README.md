# One-pass training on MNIST

This experiment reproduces Figure 3 of the paper. A four-layer LeNet-style
convolutional network with 54,314 trainable parameters processes each MNIST
training image exactly once. Before the batch-size-one SGD update at iteration
`i`, the code records the training cross-entropy score and evaluates the same
predictable score on fixed test observations.

## Paper configuration

- training observations: `n=60000`
- fixed held-out test observations: `M=1000`
- random seed: `2026`
- batch size: `1`
- step size: `eta_i = 50 / (5000 + i)`
- nominal levels: `0.50,0.55,...,0.95`
- evolution levels: `alpha=0.4,0.2,0.1,0.05`
- device: automatic CUDA selection when available, otherwise CPU

The test-set frequencies condition on one realized training stream and model
initialization, as described in the paper.

## Run

```bash
python -m pip install -r requirements.txt
bash run_experiment.sh
```

MNIST is downloaded to `data` on the first run. For a small installation
check:

```bash
python main.py --n 20 --M 10 --output-dir output-smoke
```

Use `--no-download` when the dataset already exists and network access should
not be attempted.

## Output

The `output` folder contains only figures:

- `fig3-1.{png,pdf}`: end-of-stream hold-out coverage versus nominal level.
- `fig3-2.{png,pdf}`: hold-out coverage along the training stream.

