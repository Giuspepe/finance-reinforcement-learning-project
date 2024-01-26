# Reinforcement Learning in Quantitative Finance

This project explores the application of Reinforcement Learning (RL) methodologies in the field of quantitative finance. The focus is on the empirical evaluation of RL techniques, specifically in the context of automated stock trading. We have implemented and tested two key algorithms: Imitative Recurrent Deterministic Policy Gradient (iRDPG) and Transformer Actor-Critic with Regularization (TACR), using a custom gym environment designed for single-stock trading on a daily timeframe.

## Project Overview

The project aims to evaluate the effectiveness of RL in trading scenarios with high-dimensional state spaces and partial observability, which are common in financial markets. We address the challenges of applying standard RL techniques in finance, such as the handling of out-of-distribution states and the non-stationarity of financial markets.

## Custom Trading Environment

Our custom trading environment, `SimpleOneStockStockTradingBaseEnv`, simulates realistic stock trading activities where an agent can execute discrete actions (buy, hold, sell) based on daily stock data and technical indicators. The environment is designed to provide a simplified yet realistic representation of the stock market for the RL agents to interact with.

## Algorithms

### Recurrent Deterministic Policy Gradient (RDPG) Implementation

This project contains an implementation of the Recurrent Deterministic Policy Gradient (RDPG) algorithm, an extension of the Deterministic Policy Gradient algorithm designed for environments with partial observability. 

[DDPG Paper](https://arxiv.org/pdf/1509.02971.pdf)
[RDPG Paper](https://rll.berkeley.edu/deeprlworkshop/papers/rdpg.pdf)

RDPG implementation based largely on [off-policy-continuous-control](https://github.com/zhihanyang2022/off-policy-continuous-control.git)


### Imitative Recurrent Deterministic Policy Gradient (iRDPG)

iRDPG is an algorithm that combines imitation learning with Recurrent Deterministic Policy Gradient (RDPG) to address the challenges of representing noisy financial data and balancing exploration and exploitation in trading strategies.

[iRDPG Paper](http://staff.ustc.edu.cn/~qiliuql/files/Publications/Yang-Liu-AAAI2020.pdf)


### Transformer Actor-Critic with Regularization (TACR)

This project contains an implementation of the Transformer Actor-Critic with Regularization (TACR) algorithm.

TACR leverages the Transformer architecture to capture temporal dependencies in financial data, combined with an Actor-Critic framework. It incorporates offline learning and behavior cloning regularization to learn from suboptimal trajectories efficiently.

[TACR Paper](https://www.researchgate.net/publication/374722752_Offline_Reinforcement_Learning_for_Automated_Stock_Trading)

TACR implementation based largely on [tacr](https://github.com/VarML/TACR/)
