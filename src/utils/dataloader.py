from pathlib import Path

import glob
import os
import os.path
import cv2
import re
from natsort import natsorted

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

"""
loader_pipeline: integrate the data file loading, the training set loading, test set loading and dataloader.
整合了从文件路径获取，到DataLoad 对象的创建，再到通过dataload对象分割数据集的过程，最终loader可以被用来加载数据。
"""


def loader_pipeline(train_video, train_label, test_video, test_label,
                    imgz=(25, 25), window_size=100, batch_size=10, time_delay=1, win_stride=2):
    file_paths = DataFileLoad(train_video=train_video, train_label=train_label, test_video=test_video,
                              test_label=test_label)

    ## Load label content and video frames from file path
    train_video, train_label = DataLoad(video_path=file_paths['train_video'], label_path=file_paths['train_label'],
                                        imgz=imgz)
    test_video, test_label = DataLoad(video_path=file_paths['test_video'], label_path=file_paths['test_label'],
                                      imgz=imgz)

    ## convert to Dataloader (DataSets)
    train_set = VideoFrameDataset(train_video, train_label, window_size, time_delay=time_delay, stride=win_stride)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = VideoFrameDataset(test_video, test_label, window_size=window_size, time_delay=time_delay, stride=None)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader


"""
识别目标文件夹下的视频文件和label 文件，并校对所有文件名，分别将文件名，文件路径存储到dictionary中，以供后续加载目标文件寻找路径
input: train_video, train_label, test_video, test_label are the root path or relative path for the folder of train(test) data
output: files: A dictionary representation for all train/test data path.
Parse, load the file path, and segment the data into training set and test set.
解析，加载文件路径，分割训练集和测试集
"""


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

初步处理，并加载所有数据到内存中以供后续数据的使用和分类，存储在字典中
"""


def DataLoad(video_path, label_path, imgz):
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


"""  !!!! Main Dataset loader
Dataset 类的继承，用于构建加载之前所有准备好的数据，并构建 getitem 用于后续循环获取每一个一个batch 中包含的数据
the inheritance of Dataset object, for dataset construction. And provide an abstract method for data iteration.
make sure every item giving the data contained in one batch.
"""


class VideoFrameDataset(Dataset):
    def __init__(self, video_frames_dict, labels_dict, window_size, time_delay, stride):
        self.stride = stride
        self.video_frames_dict = video_frames_dict
        self.labels_dict = labels_dict
        self.window_size = window_size
        self.time_delay = time_delay
        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []
        for labelName, (filename, frames) in zip(self.labels_dict, self.video_frames_dict.items()):
            num_frames = len(frames)
            window_size = self.window_size if self.window_size else num_frames
            stride = self.stride if self.stride else 1
            time_delay = self.time_delay if self.time_delay else 1
            length = num_frames - window_size - time_delay + 1 if time_delay >= 0 else num_frames - window_size + 1
            for start in range(0, length, stride):
                end = start + window_size
                samples.append((labelName, filename, start, end))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        labelName, fileName, start, end = self.samples[idx]
        frames = self.video_frames_dict[fileName][start:end, :, :]  # [window_size, 1, 25, 25]
        labels = self.labels_dict[labelName][end + self.time_delay]
        return frames, labels


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


# ================================#================================#================================#===================
# ============# previous dataloader only working on simple data input, the /BioLearning/RawData/InputVideo #============
# ================================#================================#================================#===================


class DragonflyDataset(Dataset):
    def __init__(self, video_path, description_path, label_path, window_size, time_delay, stride):
        self.video_path = video_path
        self.description_path = description_path
        self.label_path = label_path
        self.window_size = window_size
        self.time_delay = time_delay
        self.stride = stride

        self.frames, self.frame_count = self.load_video(self.video_path)
        self.load_descriptors()
        self.load_label()

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
            binary_frame = binary_frame // 255
            frames.append(binary_frame)

        cap.release()
        frames = np.array(frames)

        return frames, frame_count

    def load_descriptors(self):
        self.descriptors = []
        try:
            with open(self.description_path, 'r') as file:
                for line in file:
                    values = list(map(float, line.strip().split()))
                    self.descriptors.append(values)
            self.descriptors = np.array(self.descriptors)
            return True, "Discriptor loaded successfully"
        except FileNotFoundError:
            return False, "File not found."
        except PermissionError:
            return False, "Permission denied."
        except Exception as e:
            return False, f"An error occurred: {str(e)}"

    def load_label(self):
        self.labels = []
        try:
            with open(self.label_path, 'r') as file:
                content = file.readlines()
                self.labels = [int(float(line.strip())) for line in content]
            return True, "Labels loaded successfully."
        except FileNotFoundError:
            return False, "File not found."
        except PermissionError:
            return False, "Permission denied."
        except Exception as e:
            return False, f"An error occurred: {str(e)}"

    def __len__(self):
        return max(0, ((self.frame_count - self.time_delay - self.window_size) // self.stride) + 1)

    def __getitem__(self, idx):

        start_frame_idx = idx * self.stride
        end_frame_idx = start_frame_idx + self.window_size - 1
        if end_frame_idx + self.time_delay >= self.frame_count:
            raise IndexError("Index exceeds frame count")

        frames = self.frames[start_frame_idx:end_frame_idx + 1]
        descriptors = self.descriptors[start_frame_idx:end_frame_idx + 1]

        frames = np.array(frames)
        descriptors = np.array(descriptors)
        label = self.labels[end_frame_idx + self.time_delay]

        return torch.tensor(frames, dtype=torch.float32), torch.tensor(descriptors, dtype=torch.float32), label


def path_loading(ROOT):
    # 检查根目录是否存在
    if not os.path.exists(ROOT):
        print(f"Root path '{ROOT}' does not exist.")
        return None, None, None
    else:
        # 列出根目录下的所有文件和目录
        print("Contents of the root directory:", os.listdir(ROOT))

    # 读取视频文件路径
    Video_Path = glob.glob(os.path.join(ROOT, "HighFPSCut", "*.avi"))

    # 检查HighFPSVideos目录是否存在
    highfpsvideos_path = os.path.join(ROOT, "HighFPSCut")
    if not os.path.exists(highfpsvideos_path):
        print(f"HighFPSVideos path '{highfpsvideos_path}' does not exist.")
    else:
        Video_Path = natsorted(Video_Path)
        print("Contents of the HighFPSVideos directory: ", Video_Path[0], "......")

    # 读取手势描述文件路径
    Gesture_Descriptor_Path = glob.glob(os.path.join(ROOT, "CubeGesture", "*.txt"))

    # 检查CubeGesture目录是否存在
    cubegesture_path = os.path.join(ROOT, "CubeGesture")
    if not os.path.exists(cubegesture_path):
        print(f"CubeGesture path '{cubegesture_path}' does not exist.")
    else:
        Gesture_Descriptor_Path = natsorted(Gesture_Descriptor_Path)
        print("Contents of the CubeGesture directory:", Gesture_Descriptor_Path[0], "......")

    Spike_Trans_Path = glob.glob(os.path.join(ROOT, "SpikeTrainDelay", "*.txt"))

    # 检查SpikeTime目录是否存在
    spiketime_path = os.path.join(ROOT, "SpikeTrainDelay")
    if not os.path.exists(spiketime_path):
        print(f"SpikeTime path '{spiketime_path}' does not exist.")
    else:
        Spike_Trans_Path = natsorted(Spike_Trans_Path)
        print("Contents of the SpikeTime directory: ", Spike_Trans_Path[0], "......")

    return Video_Path, Gesture_Descriptor_Path, Spike_Trans_Path


def index_extractor(path):
    filename = path.split('/')[-1]
    file_idx = int(re.search(r'\d+', filename).group(0))
    return file_idx


def count_lines_in_file(path):
    try:
        with open(path, 'r') as file:
            lines = file.readlines()
            return len(lines)
    except FileNotFoundError:
        return False, "File not found."
    except PermissionError:
        return False, "Permission denied."
    except Exception as e:
        return False, f"An error occurred: {str(e)}"


def matching_check(video_path, descriptor_path, label_path):
    video_file_idx = index_extractor(video_path)
    descr_file_idx = index_extractor(descriptor_path)
    label_file_idx = index_extractor(label_path)
    if video_file_idx != descr_file_idx or descr_file_idx != label_file_idx:
        print("Index Matching Error!")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return False
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    descriptor_length = count_lines_in_file(descriptor_path)
    label_length = count_lines_in_file(label_path)
    if label_length != descriptor_length or descriptor_length != frame_length:
        print("Length Matching Error!")
        return False
    return True
