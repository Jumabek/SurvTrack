import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QGridLayout, QLabel, QPushButton, QComboBox

from track import ObjectTracker


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, video_path, tracker_name):
        super().__init__()
        self.video_path = video_path
        self.tracker_name = tracker_name

    def run(self):
        tracker = ObjectTracker(self.tracker_name)
        video = cv2.VideoCapture(self.video_path)
        if not video.isOpened():
            raise Exception("Could not open video file")
        _, frame = video.read()
        while True:
            if frame is not None:
                success, boxes = tracker.update(frame)
                if success:
                    for box in boxes:
                        (x, y, w, h) = [int(v) for v in box]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                self.change_pixmap_signal.emit(frame)
            else:
                break
            _, frame = video.read()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.video_thread = None

        self.central_widget = QLabel()
        self.setCentralWidget(self.central_widget)

        grid_layout = QGridLayout(self.central_widget)

        self.label_video = QLabel("No video file selected")
        self.label_video.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.label_video, 0, 0, 1, 2)

        self.label_tracker = QLabel("Tracker:")
        grid_layout.addWidget(self.label_tracker, 1, 0)

        self.comboBox_tracker = QComboBox()
        self.comboBox_tracker.addItem("CSRT")
        self.comboBox_tracker.addItem("KCF")
        self.comboBox_tracker.addItem("MOSSE")
        self.comboBox_tracker.addItem("MedianFlow")
        self.comboBox_tracker.addItem("TLD")
        self.comboBox_tracker.addItem("Boosting")
        self.comboBox_tracker.addItem("MIL")
        grid_layout.addWidget(self.comboBox_tracker, 1, 1)

        self.pushButton_open_video = QPushButton("Open Video File")
        self.pushButton_open_video.clicked.connect(self.open_video)
        grid_layout.addWidget(self.pushButton_open_video, 2, 0)

        self.pushButton_start_tracking = QPushButton("Start Tracking")
        self.pushButton_start_tracking.clicked.connect(self.start_tracking)
        grid_layout.addWidget(self.pushButton_start_tracking, 2, 1)

        self.pushButton_stop_tracking = QPushButton("Stop Tracking")
        self.pushButton_stop_tracking.clicked.connect(self.stop_tracking)
        grid_layout.addWidget(self.pushButton_stop_tracking, 3, 1)

        self.pushButton_quit = QPushButton("Quit")
        self.pushButton_quit.clicked.connect(self.quit)
        grid_layout.addWidget(self.pushButton_quit, 3, 0)

        self.setWindowTitle("Object Tracker")

    def open_video(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi)")
        file_dialog.setDefaultSuffix("mp4")
        if file_dialog.exec_() == QFileDialog.Accepted:
            self.video_path = file_dialog.selectedFiles()[0]
            self.label_video.setText(self.video_path)


    def start_tracking(self):
        self.tracker_name = self.comboBox_tracker.currentText()
        if self.video_thread is not None and self.video_thread.isRunning():
            return
        try:
            self.video_thread = VideoThread(self.video_path, self.tracker_name)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def stop_tracking(self):
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.terminate()

    def update_image(self, frame):
        q_image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.label_video.setPixmap(QPixmap.fromImage(q_image))

    def quit(self):
        self.stop_tracking()
        self.close()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, video_path, tracker_name):
        super().__init__()
        self.video_path = video_path
        self.tracker_name = tracker_name

    def run(self):
        tracker = ObjectTracker(self.tracker_name)
        video = cv2.VideoCapture(self.video_path)
        if not video.isOpened():
            raise Exception("Could not open video file")
        _, frame = video.read()
        while True:
            if frame is not None:
                success, boxes = tracker.update(frame)
                if success:
                    for box in boxes:
                        (x, y, w, h) = [int(v) for v in box]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                self.change_pixmap_signal.emit(frame)
            else:
                break
            _, frame = video.read()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
