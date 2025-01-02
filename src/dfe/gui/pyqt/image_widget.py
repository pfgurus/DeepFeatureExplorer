import sys

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
from dfe.utils import np2qpixmap
from PyQt5.QtCore import pyqtSignal

class ImageDisplayWidget(QLabel):
    mouse_moved = pyqtSignal()
    crop_bbox_updated = pyqtSignal()
    def __init__(self, mouse_tracking=False, save_crop_bbox=False):
        super().__init__()
        self.setScaledContents(True)  # Enable stretching of the image
        self.setAlignment(Qt.AlignCenter)  # Center the image
        self.setMouseTracking(mouse_tracking)  # Enable mouse tracking
        self.pixel_info_text = ""

        # For getting crop bbox
        self.save_crop_bbox = save_crop_bbox
        self.crop_bbox = None
        self.start_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.save_crop_bbox:
                self.set_crop_bbox(self.start_pos, event.pos())
                self.update_canvas()
                self.crop_bbox_updated.emit()

    def reset_crop_bbox(self):
        self.crop_bbox = None
        self.update_canvas()
        self.crop_bbox_updated.emit()

    def mouseMoveEvent(self, event):
        """Capture mouse movement over the widget."""
        if self.image is not None:
            pos = event.pos()
            x, y = pos.x(), pos.y()
            scaled_x, scaled_y = self.get_pixel_coordinates(x, y)
            H, W = self.image.shape[:2]

            if 0 <= scaled_x < W and 0 <= scaled_y < H:
                self.pixel_info_text = f"Pixel: ({scaled_x}, {scaled_y}) | Value: {self.image[scaled_y, scaled_x]}"
                self.mouse_moved.emit()

        if self.save_crop_bbox:
            self.set_crop_bbox(self.start_pos, event.pos())
            self.update_canvas()


    def set_crop_bbox(self, start_pos, end_pos):
        x1, y1 = self.get_pixel_coordinates(start_pos.x(), start_pos.y())
        x2, y2 = self.get_pixel_coordinates(end_pos.x(), end_pos.y())

        if x2 > x1 and y2 > y1:
            if 0 <= x1 < self.image.shape[1] and 0 <= y1 < self.image.shape[0] and 0 <= x2 < self.image.shape[
                1] and 0 <= y2 < self.image.shape[0]:
                self.crop_bbox = (x1, y1, x2, y2)


    def get_pixel_coordinates(self, x, y):
        """Get the pixel coordinates of the scaled image."""
        H, W     = self.image.shape[:2]
        scaled_x = int(x * (W / self.width()))
        scaled_y = int(y * (H / self.height()))
        return scaled_x, scaled_y

    def set_image(self, image: np.ndarray): # RGB 0-1 image
        self.image = image
        self.update_canvas()

    def update_canvas(self): # RGB 0-1 image
        image         = self.image.copy()
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        if self.crop_bbox is not None:
            x1, y1, x2, y2 = self.crop_bbox
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        image_resized = cv2.resize(image, (self.width(), self.height()), interpolation=cv2.INTER_NEAREST)
        self.setPixmap(np2qpixmap(image_resized))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer with Pixel Coordinates")

        # Path to the image file
        image_path = "assets/screenshot.png"  # Replace with your image file path

        # Create an ImageDisplayWidget
        self.image_display = ImageDisplayWidget(image_path)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_display)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
