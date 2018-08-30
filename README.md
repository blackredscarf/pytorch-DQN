Using pytorch to implement DQN (Deep Q Network) / DDQN (Double DQN) / Atari DDQN.

## Dependency
- python 3.6
- pytorch 0.4+
- tensorboard
- gym

## Train
DQN:
```
dqn.py --train --env CartPole-v0
```
DDQN:
```
ddqn.py --train --env CartPole-v0
```
Atari DDQN:
```
atari_ddqn.py --train --env PongNoFrameskip-v4
```
Parameters need to be manually adjusted within the file.

You can use the tensorboard to see the training.
```
tensorboard --logdir=out/CartPole-v0-run0
```

## Test
For `dqn.py`, `ddqn.py` and `atari_ddqn.py`, you use `--test` like this:
```
ddqn.py --test --env CartPole-v0 --model_path out/CartPole-v0-run23/model_best.pkl
```
It will render graphical interface.

## Result

### CartPole-v0
I trained CartPole-v0 environment with dqn and ddqn. (Blue is dqn, and orange is ddqn)

<img src="https://i.loli.net/2018/08/30/5b879eecb849e.png" width="800px">

### PongNoFrameskip-v4
Training Atari game PongNoFrameskip-v4 two million step.

<img src="https://i.loli.net/2018/08/30/5b87a04c84e7e.png" width="800px">

## Reference
- paper [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- repository [RL-Adventure](https://github.com/higgsfield/RL-Adventure)
