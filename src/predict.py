
import argparse
import os
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from Tools.common import select_device, DataLoad, DataFileLoad, VideoFrameDataset

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def run(train_video_path, train_label_path,
        test_video_path="",
        test_label_path="",
        weight=None,  ## The pretrained or trained pytorch model
        window_size=100,  ## Window Size for sampling
        batch_size=10,
        win_stride=2,  ## The length of step for sampling
        imgz=(25, 25),  ## resize image
        device="",  ## cuda device if have
        autoencoder=False,  ## Apply autoencoder
        predictor=True,  ## Apply predictor
        train=True,  ## Apply model for train
        test=False,  ## Apple model for test
        save_model=True,  ## Save model or not
        save_txt=False,  ## Save the train process as txt file
        name='exp'  ## The save path for experiment
        ):
    ## load file names
    file_paths = DataFileLoad(train_video=train_video_path, train_label=train_label_path, test_video=test_video_path,
                             test_label=test_label_path)
    print(file_paths)
    ## load file content
    train_video, train_label = DataLoad(video_path=file_paths['train_video'], label_path=file_paths['train_label'], imgz=imgz)
    test_video, test_label = DataLoad(video_path=file_paths['test_video'], label_path=file_paths['test_label'], imgz=imgz)

    ## convert to dataloader
    train_set = VideoFrameDataset(train_video, train_label, window_size, stride=win_stride)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = VideoFrameDataset(test_video, test_label, window_size, stride=win_stride)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    device = select_device(device)
    print(device)
    return


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_video_path", "-tv", type=str, default="", help="The path for test video dir")
    parser.add_argument("--test_label_path", "-tl", type=str, default="", help="The path for test label dir")
    parser.add_argument("--weight", type=str, default="", help="model path(s)")
    parser.add_argument("--window_size", "-w", type=int, default=10, help="The window size for video sampling")
    parser.add_argument("--batch_size", "-b", type=int, default=10, help="The size for data in one batch")
    parser.add_argument("--win_stride", "-s", type=int, default=2,
                        help="The step length of sliding window for video sampling")
    parser.add_argument("--imgz", "--img", "--img-size", nargs="+", type=int, default=[25],
                        help="inference size h,w")
    parser.add_argument("--device", default="mps", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--autoencoder", action="store_false", help="Apply the autoencoder")
    parser.add_argument("--predictor", action="store_true", help="Apply the predictor")
    parser.add_argument("--train", action="store_true", help="Use dataset for training model")
    parser.add_argument("--test", action="store_false", help="Use dataset for model test")
    parser.add_argument("--save_model", "--model", action="store_true", help="Store the model as .pt file")
    parser.add_argument("--save_txt", "--text", action="store_true", help="store the training process as .txt file")
    parser.add_argument("--name", "--save", type=str, default="exp", help="save results to project/save")
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
    main(opt)
