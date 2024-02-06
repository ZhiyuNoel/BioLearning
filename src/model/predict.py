import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.utils import DataLoad, DataFileLoad, VideoFrameDataset, increment_path
from LSTM import LSTMEncoder, LSTMAutoencoder, Predictor
from common import precision_cal, select_device, model_train

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def model_loader(weight, device): ## 待定
    Encoder = LSTMEncoder()
    predictor = Predictor(extractor=Encoder, device=device).to(device)  # 假设在没有GPU的环境中加载
    # 加载保存的状态字典
    state_dict = torch.load(weight, map_location=device)
    predictor.load_state_dict(state_dict)
    return predictor


def autoencoder_test(model: nn.Module, device, dataloader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():  # 在评估过程中不计算梯度
        for testImgs, _ in dataloader:  # 假设test_loader是您的测试数据加载器
            # 前向传播
            testImgs = testImgs.to(device)
            reconstructed_imgs = model(testImgs)
            # 计算损失
            test_loss += criterion(reconstructed_imgs, testImgs).item()
    # 计算平均损失
    avg_test_loss = test_loss / len(dataloader)

    return avg_test_loss


def run(train_video_path, train_label_path,
        test_video_path="",
        test_label_path="",
        weight=None,  ## The pretrained or trained pytorch model
        window_size=100,  ## Window Size for sampling
        batch_size=10,  ## batch size
        win_stride=2,  ## The length of step for sampling
        imgz=(25, 25),  ## resize image
        device="",  ## cuda device if have
        autoencode=False,  ## Apply autoencoder
        predict=True,  ## Apply predictor
        train=True,  ## Apply model for train
        test=False,  ## Apple model for test
        save=True,  ## Save model or not
        name='exp'  ## The save path for experiment
        ):
    ## load file names (relative path)
    file_paths = DataFileLoad(train_video=train_video_path, train_label=train_label_path, test_video=test_video_path,
                              test_label=test_label_path)
    ## Load label content and video frames from file path
    train_video, train_label = DataLoad(video_path=file_paths['train_video'], label_path=file_paths['train_label'],
                                        imgz=imgz)
    test_video, test_label = DataLoad(video_path=file_paths['test_video'], label_path=file_paths['test_label'],
                                      imgz=imgz)
    ## convert to Dataloader (DataSets)
    train_set = VideoFrameDataset(train_video, train_label, window_size, stride=win_stride)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = VideoFrameDataset(test_video, test_label, window_size, stride=win_stride)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    ## Set device selection
    device = select_device(device)
    ## Model Load
    learning_rate = 0.0001
    EPOCHS = 10
    if autoencode:
        if weight is not None:
            Autoencoder = torch.load(weight).to(device)
        else:
            Autoencoder = LSTMAutoencoder(device)

        if train:
            criterion = nn.MSELoss()
            optimizer = optim.Adam(Autoencoder.parameters(), lr=learning_rate)
            Autoencoder.train()
            print(f"Load model to device: {device}, Start Autoencoder training ............")
            Autoencoder, losses = model_train(model=Autoencoder, device=device, dataloader=train_loader,
                                              criterion=criterion, optimizer=optimizer, EPOCHS=EPOCHS,
                                              autoencode=True)
            avg_train_loss = np.mean(losses)
            print(f"Autoencoder training finished: Average training loss: {avg_train_loss}", end="   ")
        if test:
            test_loss = autoencoder_test(Autoencoder, device, dataloader=test_loader, criterion=criterion)
            print(f"Test loss: {test_loss}", end="   ")

        if save:
            AE_proj = ROOT / "runs/autoencoder"
            AE_save_dir = increment_path(Path(AE_proj) / name, exist_ok=False)  # increment run
            AE_save_dir.mkdir(parents=True, exist_ok=True)  # make dir
            torch.save(Autoencoder, AE_save_dir / "autoencoder.pt")
            print(f"The trained model is saved in: {AE_save_dir}/autoencoder.pt", end="")
        print("\n")

    if predict:
        if weight is not None:
            predictor = torch.load(weight).to(device)
        else:
            Encoder = LSTMEncoder()
            predictor = Predictor(extractor=Encoder, device=device).to(device=device)

        if train:
            criterion = nn.BCELoss()
            optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)
            print(f"Load model to device: {device}, Start Predictor training ............")
            predictor, train_loss = model_train(model=predictor, device=device, dataloader=train_loader,
                                                criterion=criterion, optimizer=optimizer, EPOCHS=EPOCHS)
            print(f"Autoencoder training finished: Average training loss: {np.mean(train_loss)}", end="   ")
        if test:
            score, test_loss = precision_cal(predictor, test_loader, device)
            print(f"scores for test = {score}; Average test loss = {test_loss}", end="   ")
        if save:
            Pre_proj = ROOT / "runs/predict"
            Pre_save_dir = increment_path(Path(Pre_proj) / name, exist_ok=False)  # increment run
            Pre_save_dir.mkdir(parents=True, exist_ok=True)  # make dir
            torch.save(predictor, Pre_save_dir / "predictor.pt")
            print(f"The trained model state dictionary is saved in: {Pre_save_dir}/predictor.pt", end="")
        print("\n")

    return


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_video_path", "-tv", type=str, default="", help="The path for test video dir")
    parser.add_argument("--test_label_path", "-tl", type=str, default="", help="The path for test label dir")
    parser.add_argument("--weight", default=None, help="model path(s)")
    parser.add_argument("--window_size", "-w", type=int, default=100, help="The window size for video sampling")
    parser.add_argument("--batch_size", "-bz", type=int, default=10, help="The size for data in one batch")
    parser.add_argument("--win_stride", "-stride", type=int, default=2,
                        help="The step length of sliding window for video sampling")
    parser.add_argument("--imgz", "--img", "--img-size", nargs="+", type=int, default=[25],
                        help="inference size h,w")
    parser.add_argument("--device", type=str, default="mps", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--autoencode", action="store_true", help="Apply the autoencoder")
    parser.add_argument("--predict", action="store_false", help="Apply the predictor")
    parser.add_argument("--train", action="store_false", help="Use dataset for training model")
    parser.add_argument("--test", action="store_true", help="Use dataset for model test")
    parser.add_argument("--save", "-s", action="store_false", help="Store all results: e.g.the model as .pt file")
    parser.add_argument("--name", type=str, default="exp", help="save results to project/save")
    opt = parser.parse_args()
    opt.imgz *= 2 if len(opt.imgz) == 1 else 1  # expand
    print(vars(opt))
    return opt


def main(opt):
    # check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    video_path = '../InputVideo/Train_Video'
    label_path = '../InputVideo/Train_Label'
    run(train_video_path=video_path, train_label_path=label_path, **vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    opt.save = False
    opt.autoencode = False
    opt.predict = True
    opt.test = True
    opt.train = False
    opt.weight="runs/predict/exp3/predictor.pt"
    main(opt)
