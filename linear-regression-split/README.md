# Rolling conformal versus split conformal for OLS

This experiment reproduces Figure 2 of the paper. It uses the same
minimum-norm linear-regression model as Section 4.1:

```text
X_i ~ N(0, I_d),   theta* = e_1,
Y_i = X_i^T theta* + epsilon_i,   epsilon_i ~ N(0, sigma^2).
```

At each partial sample size `n_i`, rolling conformal prediction (RoCP) uses
the predictable residual scores generated along the minimum-norm OLS training
trajectory. A burn-in `m` discards scores before index `m`.

For split conformal with training ratio `rho`, the code sets
`k_i=floor(rho*n_i)`, trains minimum-norm OLS on observations `1,...,k_i`,
freezes that predictor, and calibrates on observations `k_i+1,...,n_i`.

## Paper configuration

- terminal stream size: `n=5000`
- partial sample sizes: `n_i=100,120,...,5000`
- dimension: `d=200`
- noise standard deviation: `sigma=0.2`
- independent streams: `M=100`
- independent test observations per stream: `1`
- random seed: `2026`
- miscoverage levels: `alpha=0.4,0.2,0.1,0.05`
- RoCP burn-ins: `m=100,200,500,1000`
- split-conformal training ratios: `rho=0.2,0.4,0.6,0.8`

The plotted curves are empirical means across the 100 streams; no confidence
intervals are displayed. Prediction-set length is measured as exact Lebesgue
length. This is relevant for RoCP because its residual-score set can be a
union of intervals. The length figure uses a logarithmic vertical scale and
is truncated at 10.

## Run

```bash
python -m pip install -r requirements.txt
bash run_experiment.sh
```

The paper run is computationally expensive. For a small installation check:

```bash
python main.py --max-n 1040 --d 20 --trials 2 \
  --checkpoint-step 20 --output-dir output-smoke
```

Command-line options appended to `run_experiment.sh` override its saved paper
configuration.

## Output

The `output` folder contains only figures:

- `fig4-1.{png,pdf}`: average prediction-set length along the stream.
- `fig4-2.{png,pdf}`: empirical coverage along the stream.

These filenames match the corresponding figure assets in the Overleaf
project.
