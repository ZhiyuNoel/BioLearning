import cv2
import os.path
import numpy as np
import os
import platform
import torch
from torch.utils.data import Dataset, DataLoader


class VideoFrameDataset(Dataset):
    def __init__(self, video_frames_dict, labels_dict, window_size, stride):
        self.stride = stride
        self.video_frames_dict = video_frames_dict
        self.labels_dict = labels_dict
        self.window_size = window_size
        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []
        for filename, frames in self.video_frames_dict.items():
            num_frames = len(frames)
            for start in range(0, (num_frames - self.window_size + 1), self.stride):
                end = start + self.window_size
                base_filename = os.path.splitext(os.path.basename(filename))[0]
                # 构建新的文件路径
                labelName = os.path.join('../InputVideo/Train_Label', base_filename + '.txt')
                samples.append((labelName, filename, start, end))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        labelname, filename, start, end = self.samples[idx]

        frames = self.video_frames_dict[filename][start:end, :, :]  # [window_size, 1, 25, 25]
        labels = self.labels_dict[labelname][start:end]  # [sequence_length, num_classes]
        return frames, labels


def DataFileLoad(train_video, train_label, test_video="", test_label=""):
    video_file = ('.avi', '.mp4')
    if len(test_video) == 0 and len(test_label) == 0:
        files = {
            'all_video': [train_video + '/' + f for f in sorted(os.listdir(train_video)) if f.endswith(video_file)],
            'all_label': [train_label + '/' + f for f in sorted(os.listdir(train_label)) if f.endswith('.txt')],
            'train_video': [],
            'train_label': [],
            'test_video': [],
            'test_label': []}
        for index, (video, label) in enumerate(zip(files['all_video'], files['all_label'])):
            if index % 4 == 0:
                files['test_video'].append(video)
                files['test_label'].append(label)
            else:
                files['train_video'].append(video)
                files['train_label'].append(label)
        del files['all_video']
        del files['all_label']
    else:
        files = {
            'train_video': [train_video + '/' + f for f in sorted(os.listdir(train_video)) if f.endswith(video_file)],
            'train_label': [train_label + '/' + f for f in sorted(os.listdir(train_label)) if f.endswith('.txt')],
            'test_video': [test_video + '/' + f for f in sorted(os.listdir(test_video)) if f.endswith(video_file)],
            'test_label': [test_label + '/' + f for f in sorted(os.listdir(test_label)) if f.endswith('.txt')]}
    return files


def DataLoad(video_path, label_path, imgz):
    """
    Description: Preprocess of input video and label files, load the video and label files as continuous tensor files
    Parameters: video_path: The root path for videos
                video_file: The file name for each videos
                label_path: The root path for labels
                label_file: The file name for each labels
                imgz: The size for frame resize
    output: videos: A dictionary take the file name as key and tensor list stored each frame as value;
            spike_times: A dictionary take the label file name as key and a tensor list maintaining one-hot list for
                         spiking frames
    """
    ## Load Videos with frames
    width, height = imgz
    videos = {}
    spike_times = {}
    for videoName, labelName in zip(video_path, label_path):
        num_frames = 0
        videoLoader = cv2.VideoCapture(videoName)
        frame_tensors = []
        while True:
            num_frames += 1
            ret, frame = videoLoader.read()
            if ret is False:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 1. 将BGR图像转换为灰度图
            frame_gray = frame_gray.astype(np.float32) / 255.0  # 2. 将数据类型转换为float32，并缩放到0-1范围
            frame_resized = cv2.resize(frame_gray, (width, height))  # 3. 重新调整图像尺寸（如果需要保持原尺寸，这一步可以省略）
            frame_tensor = torch.tensor(frame_resized).unsqueeze(0)
            frame_tensors.append(frame_tensor)
        videos[videoName] = torch.stack(frame_tensors)  ## shape: [1, 25, 25] [channel, width, height]
        videoLoader.release()

        spikes = [0] * num_frames
        with open(labelName, 'r') as file:
            for line in file:
                try:
                    spike_index = int(line.strip())
                    spikes[spike_index] = 1
                except ValueError:
                    print(f"Skipping invalid number: {line.strip()}")
        spike_tensor = torch.tensor([element for element in spikes])
        spike_times[labelName] = spike_tensor

    return videos, spike_times


def select_device(device="", batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f"BioLearning  Python-{platform.python_version()} torch-{torch.__version__} "
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = "cuda:0"
    elif mps and torch.backends.mps.is_built() and torch.backends.mps.is_available():  # prefer MPS if available
        s += "MPS\n"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    return torch.device(arg)
