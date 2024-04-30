# Understanding RL Vision with CoinRun

Generate interfaces for interpreting vision models trained using RL.

The core utilities used to compute feature visualization, attribution and dimensionality reduction can be found in `lucid.scratch.rl_util`, a submodule of [Lucid](https://github.com/tensorflow/lucid/). These are demonstrated in [this notebook](https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/misc/rl_util.ipynb). The code here leverages these utilities to build HTML interfaces similar to the above demo.

![](https://openaipublic.blob.core.windows.net/rl-clarity/attribution/demo.gif)

## Installation

Supported platforms: MacOS and Ubuntu, Python 3.7, TensorFlow <= 1.14

- Install [Baselines](https://github.com/openai/baselines) and its dependencies, including TensorFlow 1.
- Clone the repo:
    ```
    git clone https://github.com/aptgetnitin/coinrun-deeprob-w24.git
    ```
- Install the repo and its dependencies, among which is a pinned version of [Lucid](https://github.com/tensorflow/lucid):
    ```
    pip install -e understanding-rl-vision
    ```
- Install an RL environment of your choice. Supported environments:
    - [CoinRun](https://github.com/openai/coinrun) (the original version used in the paper): follow the instructions. Note: due to CoinRun's requirements, you should re-install Baselines after installing CoinRun.

## Generating interfaces

The main script processes checkpoint files saved by RL code:
```
from understanding_rl_vision import rl_clarity

rl_clarity.run('path/to/checkpoint/file', output_dir='path/to/directory')
```

An example checkpoint file can be downloaded [here](https://openaipublic.blob.core.windows.net/rl-clarity/attribution/models/coinrun.jd), or can be generated using the [example script](understanding_rl_vision/rl_clarity/example.py). 
