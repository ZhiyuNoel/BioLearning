import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Encoding Layers
        self.Encode0 = nn.Linear(625, 100)
        self.Encode1 = nn.Linear(100, 30)
        self.Bottleneck = nn.LayerNorm(30)    #Edit these layer definitions except for the Bottleneck.
        # Decoding Layers
        self.Decode1 = nn.Linear(30, 100)
        self.Decode0 = nn.Linear(100, 625)

        self.Predict0 = nn.Linear(30, 20)
        self.Predict1 = nn.Linear(20, 20)
        self.Predict2 = nn.Linear(20, 1)

    def encode(self, x):    #Edit this method without changing the intup and output tensor dimentions.
        x = torch.tanh(self.Encode0(x))
        x = torch.sigmoid(self.Encode1(x))
        x = self.Bottleneck(x)
        return x

    def decode(self, x):    #Edit this method without changing the intup and output tensor dimentions.
        x = torch.sigmoid(self.Decode1(x))
        x = torch.tanh(self.Decode0(x))
        return x

    def Reshape_for_convolutional_layer(x, height, width):
        batch_size = 0
        if len(x.shape) == 1:
            batch_size = 1
        else:
            batch_size, _ = x.shape
        x = x.view(batch_size, 1, height, width)

    def predictor_train(self, x):    #Edit this method without changing the intup and output tensor dimentions.
        x = torch.sigmoid(self.Predict0(x))
        x = torch.relu(self.Predict1(x))
        x = torch.sigmoid(self.Predict2(x))
        return x

    '''
    串联编码与解码过程，形成自编译器
    '''
    def autoencoder_train(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    # 它首先对输入x的每个元素进行编码，然后将编码后的结果连续串联起来，最后通过预测器进行预测。
    def full_model(self, x):
        y = torch.empty(0)
        for i in range(100):
            y = torch.cat((y, (self.encode((x[i]).view(1, 625))).view(30)), dim=0)
        y = torch.flatten(y)
        x = self.predictor_train(y)
        return x



