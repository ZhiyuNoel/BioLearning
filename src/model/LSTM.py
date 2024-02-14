import torch
import torch.nn as nn
from torch.nn import functional as F

'''
========================================================================================================================
This process contains three main components including: LSTM Encoder, LSTM Decoder and LSTM Autoencoder
they are constructed with the simple LSTM layers
LSTMEncoder is able to reduce the features into 128 dimensions
LSTMDecoder reproduce images with the LSTM layer
LSTMAutoencoder combines decoder and encoder to implement the whole process
'''


class LSTMEncoder(nn.Module):
    def __init__(self, input_size=625, hidden_size=128, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        batch, seq_len, channel, width, height = x.size()
        input_size = channel * width * height
        x = x.view(batch, seq_len, input_size)
        output, (hn, cn) = self.lstm(x)
        return output, (hn, cn)


import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(self, seq_len=5, hidden_size=128, output_size=625, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        # 扩展hn成序列所需的全连接层
        self.fc_expand = nn.Linear(self.hidden_size, self.hidden_size * self.seq_len)
        # 修改后的LSTM层
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=output_size, num_layers=num_layers, batch_first=True)

        # 为了重构原始形状，可能需要转置卷积层或其他结构
        # self.conv_transpose = ...

    def forward_hn(self, hn):
        # hn形状：[num_layers * num_directions, batch_size, hidden_size]
        # 选择最后一层的hn
        hn = hn[-1]  # 形状：[batch_size, hidden_size]
        # 扩展hn成序列
        hn_seq = self.fc_expand(hn)  # 形状：[batch_size, hidden_size * seq_len]
        hn_seq = hn_seq.view(-1, self.seq_len, self.hidden_size)  # 形状：[batch_size, seq_len, hidden_size]
        # LSTM层
        output, _ = self.lstm(hn_seq)  # 输出形状：[batch_size, seq_len, output_size]
        # 根据需要调整输出形状以匹配原始输入形状
        batch_size, seq_len, _ = output.size()
        output = output.view(batch_size, seq_len, 1, 25, 25)
        # 最终输出形状应为[batch_size, seq_len, channels, height, width]
        return output

    def forward(self, x):
        # x形状：[batch, seq_len, hidden_size]
        batch, seq_len, _ = x.size()
        # LSTM层
        output, _ = self.lstm(x)
        output = output.view(batch, seq_len, 1, 25, 25)  # 调整形状以匹配目标
        return output


class LSTMAutoencoder(nn.Module):
    def __init__(self, device):
        super(LSTMAutoencoder, self).__init__()
        self.device = device
        self.encoder = LSTMEncoder().to(device=self.device)
        self.decoder = LSTMDecoder().to(device=self.device)

    def forward(self, x):
        # 编码
        output, (hn, cn) = self.encoder(x)  # 仅使用编码器的最后一个隐藏状态和单元状态
        # 解码
        # 由于解码器的输入是解码器的隐藏状态，我们需要将hn作为解码器的初始输入
        # 在实际应用中，可能需要调整维度或使用额外的全连接层来适配解码器的输入要求
        decoded = self.decoder.forward(output)
        return decoded


'''
========================================================================================================================
This process contains three main components including: LSTMCovEncoder, LSTMCovDecoder and LSTMCOvAutoencoder
they are constructed with the simple LSTM layers and convolutional layers
LSTMEncoder is able to reduce the features into 128 dimensions
LSTMDecoder reproduce images with the LSTM layer
LSTMAutoencoder combines decoder and encoder to implement the whole process
'''


class LSTMCovEncoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=32, hidden_size=128, num_layers=1):
        super(LSTMCovEncoder, self).__init__()
        self.out_channel = out_channel
        self.encodeSeq = nn.Sequential(nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=3, stride=2),
                                       nn.Conv2d(16, out_channel, kernel_size=3, stride=1, padding=1),
                                       nn.Sigmoid(),
                                       nn.MaxPool2d(kernel_size=2, stride=2))
        self.lstm_input_size = out_channel * 6 * 6  # 假设经过卷积和池化后的特征维度
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)

    def forward(self, x):
        batch, seq_len, channel, width, height = x.size()
        c_out = x.view(batch * seq_len, channel, height, width)  #
        c_out = self.encodeSeq(c_out)
        c_out = c_out.view(batch, seq_len, -1)
        output, (hn, cn) = self.lstm(c_out)
        return output, (hn, cn)


class LSTMCovDecoder(nn.Module):
    def __init__(self, in_channel=32, hidden_size=128, num_layers=1, output_channel=1):
        super(LSTMCovDecoder, self).__init__()
        self.output_channel = output_channel
        self.hidden_size = hidden_size
        # 反向LSTM层
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=in_channel * 6 * 6, num_layers=num_layers,
                            batch_first=True)

        # 反向卷积操作序列
        self.decodeSeq = nn.Sequential(
            nn.ConvTranspose2d(in_channel, 16, kernel_size=2, stride=2),  # 上采样
            nn.ReLU(),
            nn.ConvTranspose2d(16, output_channel, kernel_size=3, stride=2),  # 再次上采样
            nn.Sigmoid()  # 用Sigmoid作为最后一层，如果您的输入图像被归一化到[0,1]
        )

    def forward(self, x):
        # x形状：[batch, seq_len, hidden_size]
        batch, seq_len, _ = x.size()
        # LSTM层
        output, _ = self.lstm(x)
        # 将LSTM输出调整为合适的形状以匹配反向卷积层
        output = output.contiguous().view(batch * seq_len, -1, 6, 6)
        # 应用反向卷积序列
        output = self.decodeSeq(output)
        # 调整输出形状以匹配原始输入
        output = output.view(batch, seq_len, self.output_channel, output.size(-2), output.size(-1))
        return output


class LSTMCovAutoencoder(nn.Module):
    def __init__(self):
        super(LSTMCovAutoencoder, self).__init__()
        self.encoder = LSTMCovEncoder(in_channel=1, out_channel=32, hidden_size=128, num_layers=1)
        self.decoder = LSTMCovDecoder(in_channel=32, hidden_size=128, num_layers=1, output_channel=1)

    def forward(self, x):
        c_out, _ = self.encoder.forward(x)
        decoded = self.decoder.forward(c_out)
        return decoded


'''
========================================================================================================================
The following process concentrate on construct linear autoencoder
Main components: The Linear Encoder, Linear Decoder and a Linear Autoencoder
'''


class LinearEncoder(nn.Module):
    def __init__(self, input_size=625, hidden_size=128, latent_size=16):
        super(LinearEncoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_size)

    def forward(self, x):  # x: bs,input_size
        x = F.relu(self.linear1(x))  # -> bs,hidden_size
        x = self.linear2(x)  # -> bs,latent_size
        return x


class LinearDecoder(nn.Module):
    def __init__(self, latent_size=16, hidden_size=128, output_size=625):
        super(LinearDecoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):  # x:bs,latent_size
        x = F.relu(self.linear1(x))  # ->bs,hidden_size
        x = torch.sigmoid(self.linear2(x))  # ->bs,output_size
        return x


class LinearAutoencoder(nn.Module):
    def __init__(self, input_size=625, output_size=625, latent_size=16, hidden_size=128):
        super(LinearAutoencoder, self).__init__()
        self.encoder = LinearEncoder(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size)
        self.decoder = LinearDecoder(latent_size=latent_size, hidden_size=hidden_size, output_size=output_size)

    def forward(self, x, device):  # x: bs,input_size
        self.encoder.to(device=device)
        self.decoder.to(device=device)
        batch, seq_len, channel, width, height = x.size()
        c_out = x.view(batch * seq_len, channel * height * width)  #
        feat = self.encoder(c_out)  # feat: bs,latent_size
        re_x = self.decoder(feat)  # re_x: bs, output_size
        out_x = re_x.view(batch, seq_len, channel, width, height)
        return out_x


'''
========================================================================================================================
Predictor
'''


class Predictor(nn.Module):
    num_class = 1
    hidden_size1 = 64
    hidden_size2 = 32

    def __init__(self, extractor, device, feature_size=128, num_class=1):
        super(Predictor, self).__init__()
        self.device = device
        self.extractor = extractor.to(device=self.device)
        self.num_class = num_class

        self.LSTM_classifier = nn.LSTM(feature_size, self.hidden_size1, 1, batch_first=True)  ## 用lstm layer 处理序列，输出为64
        self.fc_1 = nn.Linear(self.hidden_size1, self.hidden_size2)  ## 用全链接层分类 full connection layer 1
        self.fc_2 = nn.Linear(self.hidden_size2, self.num_class)  ## full connection layer 2

    def forward(self, x):  ## input [batch_size, seq_len, 128]
        x_out = self.extractor(x)
        if len(x_out) != 1:
            x_out = x_out[0]
        batch_size, seq_len, features = x_out.size()
        out, _ = self.LSTM_classifier(x_out)  ## out = [batch_size, seq_len, 64]
        out = out.contiguous().view(-1, self.hidden_size1)  ## out = [batch_size * seq_len, 64]
        out = self.fc_1(out)  # 中间层激活 out = [batch_size * seq_len, 32]
        out = self.fc_2(out)  ## out = [batch_size * seq_len, 1]
        out = out.view(batch_size, seq_len, -1)  ## [10, 5, 1]
        out = torch.sigmoid(out)

        return out.squeeze(-1)


'''
========================================================================================================================
Tested, The second linear autoencoder
'''


class LinearEncoder2(nn.Module):
    def __init__(self):
        super(LinearEncoder2, self).__init__()
        # Encoding Layers
        self.Encode0 = nn.Linear(625, 100)
        self.Encode1 = nn.Linear(100, 30)
        self.Bottleneck = nn.LayerNorm(30)  # Edit these layer definitions except for the Bottleneck.

    def forward(self, x):  # x: bs,input_size
        x = torch.tanh(self.Encode0(x))
        x = torch.sigmoid(self.Encode1(x))
        x = self.Bottleneck(x)
        return x


class LinearDecoder2(nn.Module):
    def __init__(self):
        super(LinearDecoder2, self).__init__()
        # Decoding Layers
        self.Decode1 = nn.Linear(30, 100)
        self.Decode0 = nn.Linear(100, 625)

    def forward(self, x):  # x:bs,latent_size
        x = torch.sigmoid(self.Decode1(x))
        x = torch.tanh(self.Decode0(x))
        return x


class LinearAutoencoder2(nn.Module):
    def __init__(self):
        super(LinearAutoencoder2, self).__init__()
        self.encoder = LinearEncoder2()
        self.decoder = LinearDecoder2()

    def forward(self, x):  # x: bs,input_size
        batch, seq_len, channel, width, height = x.size()
        c_out = x.view(batch * seq_len, channel * height * width)  #
        feat = self.encoder(c_out)  # feat: bs,latent_size
        re_x = self.decoder(feat)  # re_x: bs, output_size
        out_x = re_x.view(batch, seq_len, channel, width, height)
        return out_x
