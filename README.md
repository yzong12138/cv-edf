### Experimental setup

1. Install `uv` [here](https://docs.astral.sh/uv/getting-started/installation/). 
2. Install the package by `uv sync`. It will install the same package as in [uv.lock](uv.lock)
3. Running the single experiment by using
```unix
uv run python main.py --model b0 --lr 3e-4
```
4. Running a batch of experiments: Change first the hyperparameters first then using 
```unix
bash launch_sweep.sh
```