# NetEvolve

## Changelog

`optim.py` and `rl.py` are very dulipcate code, but `optim.py` is for the optimization of hyperparameter and `rl.py` is for the training of the RL model. These two files are merged into `main.py`, with a command line option to select the mode.

`config.py` is removed and the configuration is moved to `main.py`, by command line options. `init_real_data.py` is moved `data_loader.py`.

## Run
```
python optimize_reward.py
python main.py --mode run
```