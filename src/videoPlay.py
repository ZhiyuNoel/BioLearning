import time
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
import cv2


class VideoWidget(QWidget):
    VIDEO_TYPE_OFFLINE = 0
    VIDEO_TYPE_REAL_TIME = 1

    NONFILE_URL = "Images/Background.png"

    desired_width = 480  # 期望的宽度
    desired_height = 360  # 期望的高度

    STATU_INIT = 0
    STATUS_PLAYING = 1
    STATU_PAUSE = 2

    video_url = ''

    def __init__(self, video_url="", video_type=VIDEO_TYPE_OFFLINE, auto_play=False):
        super().__init__()
        self.video_url = video_url
        self.video_type = video_type
        self.auto_play = auto_play
        self.status = self.STATU_INIT

        self.pictureLabel = QLabel()

        self.playButton = QPushButton()

        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.playButton.clicked.connect(self.switch_video)

        control_box = QHBoxLayout()
        control_box.setContentsMargins(0, 0, 0, 0)
        control_box.addWidget(self.playButton)

        layout = QVBoxLayout()
        layout.addWidget(self.pictureLabel)
        layout.addLayout(control_box)

        self.setLayout(layout)

        # timer 定时器
        self.timer = VideoTimer()
        self.timer.timeSignal.signal[str].connect(self.show_video_images)

        # video 捕获设置
        self.playCapture = cv2.VideoCapture()
        if self.video_url != "":
            self.set_timer_fps()
            if self.auto_play:
                self.switch_video()
        # self.videoWriter = VideoWriter('x.mp4', VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, size)

    def reset(self):
        self.timer.stop()
        self.playCapture.release()
        self.status = VideoWidget.STATU_INIT
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def set_timer_fps(self):
        self.playCapture.open(self.video_url)
        fps = self.playCapture.get(cv2.CAP_PROP_FPS)
        self.timer.set_fps(fps)
        self.playCapture.release()

    def set_video(self, url, video_type=VIDEO_TYPE_OFFLINE, auto_play=False):
        self.reset()
        self.video_url = url
        print(self.video_url)
        self.video_type = video_type
        self.auto_play = auto_play
        self.set_timer_fps()
        if self.auto_play:
            self.switch_video()

    def play(self):
        if self.video_url == "" or self.video_url is None:
            return
        if not self.playCapture.isOpened():
            self.playCapture.open(self.video_url)
        self.timer.start()
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.status = VideoWidget.STATUS_PLAYING

    def stop(self):
        if self.video_url == "" or self.video_url is None:
            return
        if self.playCapture.isOpened():
            self.timer.stop()
        if self.video_type is VideoWidget.VIDEO_TYPE_REAL_TIME:
            self.playCapture.release()
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.status = VideoWidget.STATUS_PAUSE

    def re_play(self):
        if self.video_url == "" or self.video_url is None:
            return
        self.playCapture.release()
        self.playCapture.open(self.video_url)
        self.timer.start()
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.status = VideoWidget.STATUS_PLAYING

    def show_video_images(self):
        if self.playCapture.isOpened():
            success, frame = self.playCapture.read()
            if success:
                frame = cv2.resize(frame, (self.desired_width, self.desired_height))
                height, width = frame.shape[:2]
                if frame.ndim == 3:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif frame.ndim == 2:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                temp_image = QImage(rgb.flatten(), width, height, QImage.Format.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image)
                self.pictureLabel.setPixmap(temp_pixmap)
            else:
                print("read failed, no frame data")
                success, frame = self.playCapture.read()

                if not success and self.video_type is VideoWidget.VIDEO_TYPE_OFFLINE:
                    print("play finished")  # #UJlfzasihsctHgiseee
                    self.reset()
                    self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
                return
        else:
            print("open file or capturing device error, init again")
            self.reset()

    def switch_video(self):
        if self.video_url == "" or self.video_url is None:
            return
        if self.status is VideoWidget.STATU_INIT:
            self.playCapture.open(self.video_url)
            self.timer.start()
            self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        elif self.status is VideoWidget.STATUS_PLAYING:
            self.timer.stop()
            if self.video_type is VideoWidget.VIDEO_TYPE_REAL_TIME:
                self.playCapture.release()
            self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        elif self.status is VideoWidget.STATU_PAUSE:
            if self.video_type is VideoWidget.VIDEO_TYPE_REAL_TIME:
                self.playCapture.open(self.video_url)
            self.timer.start()
            self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))

        self.status = (VideoWidget.STATUS_PLAYING,
                       VideoWidget.STATU_PAUSE,
                       VideoWidget.STATUS_PLAYING)[self.status]


class Communicate(QObject):
    signal = pyqtSignal(str)


class VideoTimer(QThread):
    def __init__(self, frequent=20):
        super().__init__()
        self.stopped = False
        self.frequent = frequent
        self.timeSignal = Communicate()
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False

        while True:
            if self.stopped:
                return
            self.timeSignal.signal.emit("1")
            time.sleep(1 / self.frequent)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps
