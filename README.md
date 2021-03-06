# Animal State Learning

This repo contains the unofficial implementation of some experiments from the paper ["From Eye-blinks to State Construction: Diagnostic Benchmarks for Online Representation Learning"](https://arxiv.org/abs/2011.04590). The code for the benchmark is taken from [here](https://github.com/banafsheh-rafiee/From-Eye-blinks-to-State-Construction-Diagnostic-Benchmarks-for-Online-Representation-Learning).


## A few instructions
`-w` to use the wandb logger instead of using a database.

`-v` verbose

1. Setup using requirements.txt
2. Default params are given in `config-defaults.yaml`. Sweep params are provided in `sweep.yaml`.

## W&B Setup

You need to initialize W&B to log the results. If it's your first time using W&B on a machine, you will need to log in:
```
$ wandb login
```
You will be asked for your API key, which appears on your W&B profile settings page.



## Reference
If you use this code in your work, you can reference the original authors at

```
@misc{rafiee2021eyeblinks,
      title={From Eye-blinks to State Construction: Diagnostic Benchmarks for Online Representation Learning}, 
      author={Banafsheh Rafiee and Zaheer Abbas and Sina Ghiassian and Raksha Kumaraswamy and Richard Sutton and Elliot Ludvig and Adam White},
      year={2021},
      eprint={2011.04590},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
