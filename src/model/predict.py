import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PyQt6.QtCore import *
from PyQt6.QtGui import QImage
from torch import nn, optim
from torch.utils.data import DataLoader

from src.model import *
from src.model.LSTM import Model
from src.utils import increment_path, loader_pipeline

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class ModelThread(QThread):
    update_progress = pyqtSignal(int)
    update_text = pyqtSignal(str)
    update_pic = pyqtSignal(QImage, QImage)

    def __init__(self, parent, device=None):
        super().__init__(parent=parent)
        self.model=None
        self.data_loader = None
        self.is_running = True
        self.paused = False
        self.device = select_device("mps" if torch.backends.mps.is_available() else "0") if device is None else device
        self.criterion = nn.MSELoss()

    def init_parameter(self, model: Model, dataloader: DataLoader):
        self.model = model.to(self.device)
        self.model.load_device(device=self.device)
        self.data_loader = dataloader
        self.is_running = True
        self.paused = False

    def set_parameters(self, *args, **kwargs):
        pass

    def stop(self):
        self.is_running = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def process_images(self, images):
        processed_images = []
        for img in images:
            img = img.cpu().detach().numpy().transpose((1, 2, 0))  # CHW to HWC
            img = (img * 255).astype(np.uint8)  # Normalize
            qimg = QImage(img, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_Grayscale8)
            processed_images.append(qimg)
        self.update_pic.emit(*processed_images)

    def get_model(self):
        return self.model

    def update_model(self, model: nn.Module):
        self.model = model


class TestThread(ModelThread):
    def run(self):
        self.model.eval()
        while self.is_running:
            with torch.no_grad():
                for testImgs, _ in self.data_loader:
                    testImgs = testImgs.to(self.device)
                    reconstructed_imgs = self.model(testImgs)
                    for tFrame, rFrame in zip(testImgs[0], reconstructed_imgs[0]):
                        while self.paused and self.is_running:
                            QEventLoop().processEvents()
                        if not self.is_running:
                            break
                        self.msleep(17)
                        self.process_images([tFrame, rFrame])

    def set_parameters(self, model: Model, data_loader: DataLoader):
        super().init_parameter(model=model, dataloader=data_loader)


class TrainingThread(ModelThread):
    def __init__(self, parent):
        super().__init__(parent)
        self.schedule = None
        self.start_lr = None
        self.optimizer = None

    def set_parameters(self, model: Model, data_loader: DataLoader, start_lr, end_lr):
        self.init_parameter(model=model, dataloader=data_loader)
        self.start_lr = start_lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=start_lr)
        self.start_lr = start_lr
        # Configure scheduler
        step_size = self.calculate_step_size(start_lr, end_lr, len(data_loader))
        gamma = 0.1 if start_lr >= end_lr else 10
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def calculate_step_size(self, start_lr, end_lr, length):
        if start_lr >= end_lr:
            return round(length / (math.log(start_lr / end_lr, 10) + 1))
        else:
            return round(length / (math.log(end_lr / start_lr, 10) + 1))

    def run(self):
        text = "Start Training: "
        EPOCH = 0
        while self.is_running:
            EPOCH += 1
            text += f" \n EPOCH {EPOCH}: "
            self.optimizer.param_groups[0]['lr'] = self.start_lr  # 重置为初始学习率
            for step, (frames, _) in enumerate(self.data_loader):
                while self.paused and self.is_running:
                    QEventLoop().processEvents()
                frames = frames.to(device=self.device)
                self.optimizer.zero_grad()
                outputs = self.model(frames)
                loss = self.criterion(outputs, frames)
                loss.backward()
                # 更新参数
                self.optimizer.step()
                self.schedule.step()
                if step % 10 == 0:
                    self.process_images([frames[0, 0], outputs[0, 0]])
                self.update_progress.emit(step)  # 发送进度信号
                self.update_text.emit(text + f"Step {step}: Loss: {loss}")  # 发送文本更新信号
                if not self.is_running:
                    break
            text += f"Step {step}: Loss: {loss}"

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
        name='exp',  ## The save path for experiment
        autoencode=False,  ## Apply autoencoder
        predict=True,  ## Apply predictor
        train=True,  ## Apply model for train
        test=False,  ## Apple model for test
        save=True  ## Save model or not
        ):
    train_loader, test_loader = loader_pipeline(train_video=train_video_path, train_label=train_label_path,
                                                test_video=test_video_path, test_label=test_label_path,
                                                window_size=window_size, win_stride=win_stride, imgz=imgz,
                                                batch_size=batch_size)

    ## Set device selection
    device = select_device(device)
    ## Model Load
    learning_rate = 0.0001
    EPOCHS = 10
    if autoencode:
        if weight is not None:
            Autoencoder = model_loader(weight, device)
        else:
            Autoencoder = LSTMAutoencoder()
        criterion = nn.MSELoss()
        if train:
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
            predictor = model_loader(weight, device=device)
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
            # get the predicted index for results:
            # post_result = result_post_processing(predictor=predictor, dataloader=test_loader, device=device)
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
    opt.weight = "runs/predict/exp3/predictor.pt"
    main(opt)
