from pathlib import Path

import cv2
import os.path
import numpy as np
import os
import torch
from torch.utils.data import Dataset


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


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path