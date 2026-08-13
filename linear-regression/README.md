# Minimum-norm linear regression

This experiment reproduces Figure 1 of the paper. Features and responses are
generated as

```text
X_i ~ N(0, I_d),   theta* = e_1,
Y_i = X_i^T theta* + epsilon_i,   epsilon_i ~ N(0, sigma^2).
```

At stream index `i`, the predictable score is the squared residual computed
with the minimum-norm least-squares estimator fitted to observations
`1,...,i-1`. The stream crosses the interpolation threshold at `i=d`.

## Paper configuration

- stream length: `n=40000`
- dimension: `d=200`
- noise standard deviation: `sigma=1`
- independent streams: `M=100`
- hold-out observations per stream: `500`
- random seed: `2026`
- nominal levels in Figure 1(a): `0.50,0.55,...,0.95`
- levels in Figure 1(b): `alpha=0.4,0.2,0.1,0.05`
- prediction-set level in Figure 1(c): `90%`
- candidate-response grid: `401` points
- evaluation stride: `1`

The four fixed features in Figure 1(c) are `sqrt(d)e_1`, `-sqrt(d)e_1`,
`sqrt(d)e_2`, and `sqrt(d/2)(e_1+e_2)`.

## Run

```bash
python -m pip install -r requirements.txt
bash run_experiment.sh
```

For a small installation check:

```bash
python main.py --n 40 --d 16 --trials 2 --holdout-size 20 \
  --y-grid-size 51 --output-dir output-smoke
```

Command-line options appended to `run_experiment.sh` override its saved paper
configuration.

## Output

The `output` folder contains only figures:

- `fig1-1.{png,pdf}`: end-of-stream empirical coverage versus nominal level.
- `fig1-2.{png,pdf}`: empirical coverage along the stream.
- `fig1-3.{png,pdf}`: fixed-feature prediction-set inclusion frequencies.

