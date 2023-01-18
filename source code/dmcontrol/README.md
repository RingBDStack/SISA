# Learning Markov State Abstractions for Deep Reinforcement Learning

This codebase was adapted from [Reinforcement Learning with Augmented Data](https://mishalaskin.github.io/rad), which was originally forked from [CURL](https://mishalaskin.github.io/curl).

See the original RAD repo for information regarding RAD.

## Installation

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

If that doesn't work, use pip to install either of the requirements files.


## Instructions
To reproduce our results use the following commands.

### Markov
```sh
# for seed in {1..10}:
#   for domain, task in [
#     ('ball_in_cup', 'catch'),
#     ('cartpole', 'swingup'),
#     ('cheetah', 'run'),
#     ('finger', 'spin'),
#     ('reacher', 'easy'),
#     ('walker', 'walk),
#   ]:
#     OTHER_SETTINGS = [check_appendix]
python -m train --replicate --markov --markov_pretrain_batch_size 512 --init_steps [INIT_STEPS] --markov_catchup_steps [CATCHUP_STEPS] --markov_pretrain_steps 100000 --markov_inv_coef [COEF_INV] --markov_smoothness_coef [COEF_SMOOTH] --markov_smoothness_max_dz 0.01 --domain_name [domain] --task_name [task] --markov_lr [LR] --seed [seed] --tag markov-agent --work_dir ./tmp/markov
```

### RAD
```sh
# for seed in range(10):
#   for domain, task in [
#     ('ball_in_cup', 'catch'),
#     ('cartpole', 'swingup'),
#     ('cheetah', 'run'),
#     ('finger', 'spin'),
#     ('reacher', 'easy'),
#     ('walker', 'walk),
#   ]:
python -m train --replicate --seed [seed] --domain_name [domain] --task_name [task] --tag rad-agent --work_dir ./tmp/rad
```

Explanation of important, non-obvious args:
```
  `--replicate` - ensures that the RAD settings replicate the original RAD paper
  `--markov`    - enables the Markov abstraction objectives. It is intended to be used
                  in conjunction with the other Markov hyperparameters.
  `--work_dir`  - where to store learning performance results
  `--tag`       - a tag to use for each experiment (mainly used during hyperparameter
                  tuning)
```

The results for each trial will be saved into an `eval.log` file, which can be loaded
into pandas as JSON data.
