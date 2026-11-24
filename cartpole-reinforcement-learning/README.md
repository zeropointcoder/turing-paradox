# CartPole Reinforcement Learning

A reinforcement learning system that learns control behaviour through repeated interaction with an environment.


## Overview
- The environment provides a state vector `s` and reward `r`

- A Deep `Q`-Network approximates the action-value function `Q(s, a)`

- Actions are selected using `ε`-greedy exploration

- Transitions `(s, a, r, s')` are stored in `replay` memory

- Network parameters are updated by minimising:
    `L = (r + γ max Q(s', a') − Q(s, a))²`

- A target network `stabilises` learning by `delayed` updates


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```