# BioLearning
The individual project for Zhiyu Dong

## Environment Setup:
1. 


## UI Development
### 日志记录
| 时间 | 目标文件 | 日志 | 状态 |
|:---|:-----|:---|:---|
|    |      |    |    |
|    |      |    |    |
|    |      |    |    |
|    |      |    |    |

## Backend Development
### 日志记录
| 时间        | 目标文件                                                                          | 日志                                                                                  | 状态                                                                                  |
|:----------|:------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
| 2024-1-29 | [原始神经网络架构](src/model/network_init.py)                                         | 完成了从BioLearning 1.0 原始架构的重构和重现，并完成模型Autoencoder的训练到测试                               | 重构，并成功完成训练，表现未达到Biolearning 1.0效果                                                   |
| 2024-2-1  | [重构神经网络](src/model/LSTM.py)<br/>[LSTM网络性能测试](src/model/LSTM_test.ipynb)       | 利用LSTM层，和线性层分别构建了LSTM-Linear Autoencoder, Linear Autoencoder 和 CNN-LSTM Autoencoder | Linear Autoencoder 与LSTM-Linear Autoencoder在目前简单的输出环境下表现相较于CNN-LSTM Autoencoder表现更好 |
| 2024-2-4  | [LSTM predictor](src/model/LSTM.py)<br/>[LSTM Pipeline](src/model/predict.py) | 构建，部署LSTM prediction pipeline, 训练当前模型并完成测试                                          | 模型在训练序列上的loss降低至0.1一下，Accuracy 达到95%， precision目前到达40%                              |
| 2024-2-7  | [LSTM predictor](src/model/LSTM.py)<br/>[LSTM Pipeline](src/model/predict.py) | 基本完成对整个序列预测算法的打包与优化，完成了从one-hot list 到 frame index 的输出                              | 模型对整个测试集序列的预测Accuracy达到99.35%，Precision达到86.82%                                     |