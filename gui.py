import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from ui_mainwindow import Ui_MainWindow
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
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.video_thread = None
        self.ui.pushButton_open_video.clicked.connect(self.open_video)
        self.ui.pushButton_start_tracking.clicked.connect(self.start_tracking)
        self.ui.pushButton_stop_tracking.clicked.connect(self.stop_tracking)
        self.ui.pushButton_quit.clicked.connect(self.quit)

    def open_video(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi)")
        file_dialog.setDefaultSuffix("mp4")
        if file_dialog.exec_() == QFileDialog.Accepted:
            self.video_path = file_dialog.selectedFiles()[0]
            self.ui.lineEdit_video_path.setText(self.video_path)

    def start_tracking(self):
        self.tracker_name = self.ui.comboBox_tracker.currentText()
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
        self.ui.label_video.setPixmap(QPixmap.fromImage(q_image))

    def quit(self):
        self.stop_tracking()
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
