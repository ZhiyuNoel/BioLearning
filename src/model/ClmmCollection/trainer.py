from torch import optim
from torch.utils.data import DataLoader

import torch.nn as nn
from tqdm import tqdm

from src.model import select_device
from src.model.ClmmCollection.CLMM import CLMM
from src.utils import DragonflyDataset, path_loading, matching_check

"""
CNN-LSTM-perceptron multiple model: 
1. The CNN layers for feature extraction from frames 
2. The LSTM layers for time-series data analysis according to the features extracted from CNN
3. The perceptron-LSTM for text analysis from descriptions of the videos
4. Then Combine the output of perceptron-LSTM and output of CNN-LSTM
"""


def main():
    window_size = 20
    batch_size = 4
    time_delay = 0
    stride = 3
    num_epochs = 10
    device = select_device("mps")

    model = CLMM(hidden_dim_img=32, num_layers=1).to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    ROOT = "../../../RawData/Dragonfly_Visual_Neuron"
    Video_Path, Descriptor_Path, Spike_Path = path_loading(ROOT)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch} in processing ...")
        model.train()
        running_loss = 0.0

        for video_path, descriptor_path, label_path in zip(Video_Path, Descriptor_Path, Spike_Path):
            if not matching_check(video_path, descriptor_path, label_path):
                print("File Matching Error!")
                continue

            dataset = DragonflyDataset(video_path, descriptor_path, label_path, window_size, time_delay, stride)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            print("Data successfully loaded!")
            for frames_batch, descriptors_batch, label in tqdm(data_loader):
                frames, desc, labels = frames_batch.to(device), descriptors_batch.to(device), label.to(device)
                optimizer.zero_grad()
                outputs = model(frames, desc)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * frames.size(0)


if __name__ == "__main__":
    main()
