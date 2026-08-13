# SGD for multiclass logistic regression

This experiment reproduces Figure 2 of the paper. It uses a five-class
softmax model with `X_i ~ N(0,I_d)` and oracle coefficient vectors

```text
theta_1* = e_1
theta_2* = e_2
theta_3* = e_3
theta_4* = (e_1 + e_2)/2
theta_5* = (e_2 + e_3 + e_4 + e_5)/2.
```

Starting from zero, the classifier processes each observation once with

```text
eta_i = eta0 / (t0 + i)^gamma.
```

The experiment compares the predictable cross-entropy score with the running
average of the signed logit margin over the most recent `T` model snapshots.

## Paper configuration

- stream length: `n=10000`
- dimension: `d=10`
- independent streams: `M=100`
- hold-out observations per stream: `500`
- random seed: `2026`
- step-size parameters: `eta0=1`, `t0=10`
- step-size exponents: `gamma=0.6,0.8,1`
- margin window: `T=100`
- nominal levels: `0.50,0.55,...,0.95`
- evolution levels: `alpha=0.4,0.2,0.1,0.05`
- fixed-feature prediction-set level: `70%` (`alpha=0.3`)
- hold-out chunk size: `100`
- margin block size: `64`

The four fixed features are `4e_1`, `-4e_1`, `4e_3`, and `3(e_4+e_5)`.

## Run

```bash
python -m pip install -r requirements.txt
bash run_experiment.sh
```

For a small installation check:

```bash
python main.py --n 40 --trials 2 --holdout-size 20 \
  --margin-window 10 --output-dir output-smoke
```

## Output

The `output` folder contains only figures:

- `fig2-1.{png,pdf}`: end-of-stream coverage, cross-entropy score.
- `fig2-2.{png,pdf}`: end-of-stream coverage, running-margin score.
- `fig2-3.{png,pdf}`: coverage paths, cross-entropy score.
- `fig2-4.{png,pdf}`: coverage paths, running-margin score.
- `fig2-5.{png,pdf}`: fixed-feature inclusion paths, cross-entropy score.
- `fig2-6.{png,pdf}`: fixed-feature inclusion paths, running-margin score.

