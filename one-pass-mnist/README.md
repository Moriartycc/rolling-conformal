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
- optimizer: pure SGD, with no momentum or weight decay
- step size: `eta_i = 50 / (5000 + i)`
- nominal levels: `0.50,0.55,...,0.95`
- evolution levels: `alpha=0.4,0.2,0.1,0.05`
- device: automatic CUDA selection when available, otherwise CPU

The four trainable layers are

1. `Conv2d(1, 8, kernel_size=5, padding=2)`, ReLU, and `MaxPool2d(2)`;
2. `Conv2d(8, 16, kernel_size=5, padding=2)`, ReLU, and `MaxPool2d(2)`;
3. flattening followed by `Linear(16*7*7, 64)` and ReLU; and
4. `Linear(64, 10)` for the class logits.

This gives 54,314 trainable parameters. Convolutional and linear weights use
Kaiming-normal initialization and all biases start at zero. Images are scaled
to `[0,1]` and normalized using the MNIST mean `0.1307` and standard deviation
`0.3081`; no data augmentation is applied. The seed fixes the model
initialization, the random order of all 60,000 training images, and the fixed
subset of 1,000 test images.

At iteration `i`, the code evaluates the training and held-out cross-entropy
scores using the parameters fitted on observations `1,...,i-1`, updates the
rolling exceedance counts, and only then performs the batch-size-one SGD
update. Thus, every score used by rolling conformal prediction is predictable
with respect to the training stream.

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
