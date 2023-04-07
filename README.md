# Tiny ChatGPT
The language model of ChatGPT is fine-tuned using reinforcement learning from human feedback (RLHF). This project is meant for researching the reinforcement learning algorithm of ChatGPT on a conceptual level.

It doesn't try to replicate the exact training process of ChatGPT.

## Toy Problem
We use a [toy problem](problem.ipynb) to analyze the training algorithm of ChatGPT.

## Differences to ChatGPT
* Use of a toy language model
* Initial policy is a random language model compared to a pre-trained language model
* No reward model, rewards are known
* Modified version of PPO without using a neural network

## PPO
ChatGPT uses the reinforcement learning algorithm proximal policy optimization (PPO) to fine-tune the language model.

### Generalized Advantage Estimation
PPO is based on [generalized advantage estimation](gae.ipynb). If there are two timesteps, then the generalized advantage estimator (GAE) is computed as follows:

```python
δ0 = R1 + γ * V(S1) - V(S0)
δ1 = R2 + γ * V(S2) - V(S1)

GAE0 = pow(γλ,0) * δ0 + 
       pow(γλ,1) * δ1
     = pow(γλ,0) * R1 + γ * V(S1) - V(S0) +
       pow(γλ,1) * R2 + γ * V(S2) - V(S1)
```

The generalized advantage estimator has a parameter `λ` which can be used to adjust the bias-variance tradeoff.
There are two special cases of the estimator, obtained by setting `λ = 0` and `λ = 1`.

```python
* λ = 0: GAE0 = pow(0,0) * δ0 + pow(0,1) * δ1 = δ0  # low bias, high variance
* λ = 1: GAE0 = pow(γ,0) * δ0 + pow(γ,1) * δ1       # high bias, low variance
```

### Differences to PPO
1. This repo implements generalized advantage estimation (GAE) but it doesn't use a neural network to estimate the state value function.
2. The policy is updated by assigning the probability `1` to the action with the highest action value. PPO uses a neural network for the policy and updates the policy by maximizing the expected value of the generalized advantage estimator (multiplied with a probability ratio).

## GAE versus Temporal Difference Learning
Generalized advantage estimation is equivalent to n-step temporal difference learning when λ is set to `1`. An example is given [here](gae_versus_td.ipynb).
