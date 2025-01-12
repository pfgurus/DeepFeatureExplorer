import os
import sys
from pathlib import Path

import cv2
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QCheckBox,
    QGridLayout, QSpinBox,
    QFileDialog
)

from dfe.common.types import RGB255
from dfe.gui.pyqt.image_widget import ImageDisplayWidget
from dfe.gui.pyqt.table_widget import SelectionTableWidget
from dfe.networks.base import UnwrappedNetwork
from dfe.utils import get_default_image, np2torch, get_feature_maps

LAYOUTS = {'H': QHBoxLayout, 'V': QVBoxLayout, 'B': QGridLayout}

class NetworkVisualizerApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Network Visualizer")
        self.setMinimumSize(1500,500)

        # Network
        self._network           = UnwrappedNetwork()

        # Internals
        self._crop_bbox         = None  # (x1, y1, x2, y2)
        self._input_img         = get_default_image()
        self._result            = {}
        self._feature_maps      = {}

        # Images to display
        self._img_in_display    = self._input_img
        self._img_feat_display  = self._input_img

        self._init_ui()
        self.display_images()

    def register_module(self, module, depth=1, name=None):
        self._network.register_module(module, depth, name)

    def set_forward(self, forward_func):
        try:
            x = torch.randn(1,3,256,256)
            out = forward_func(x)
            assert isinstance(out, dict), f'forward function should return dictionary, not {type(out)}'
        except Exception as e:
            print(e)

        self._network.set_forward(forward_func)

        # Put intermediate feature maps in table
        self.network_inference(self._input_img)
        for idx,(name, feature_map) in enumerate(self._feature_maps.items()):
            self.feature_maps_table.add_item(name, feature_map.resolution, feature_map.channels, index=idx)
        self.feature_maps_table.table.selectRow(0)

    def close(self):
        super().close()


    def _init_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Column 2: Input image display
        self.column2 = QVBoxLayout()
        self.image_in_text = QLabel("Input Image")
        self.image_in_display = ImageDisplayWidget(mouse_tracking=False, save_crop_bbox=True)
        self.image_in_display.setStyleSheet("border: 3px solid black;")
        self.image_in_display.crop_bbox_updated.connect(self.update_input)
        self.column2.addLayout(self._create_layout([(self.image_in_text, 0), (self.image_in_display, 1)], type='V'))

        # Column 3: Output image display
        self.column3 = QVBoxLayout()
        self.image_out_text = QLabel("Output/Feature Image")
        self.pixel_info_text = QLabel("Pixel Info")
        self.image_out_display = ImageDisplayWidget(mouse_tracking=True, save_crop_bbox=False)
        self.image_out_display.setStyleSheet("border: 3px solid black;")
        self.image_out_display.mouse_moved.connect(
            lambda: self.pixel_info_text.setText(self.image_out_display.pixel_info_text))
        self.column3.addLayout(self._create_layout([(self.image_out_text, 0), (self.image_out_display, 1),
                                                    (self.pixel_info_text, 0)], type='V'))

        # Column1: LEFT OPTIONS COLUMN
        self.column1 = QVBoxLayout()

        # Image/Video Input
        self.image_path_label = QLabel("Image/Video Path:")
        self.image_path_input = QLineEdit()
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load)

        # RGB, normalize buttons, channel selection spinbox
        self.rgb_button = QCheckBox('RGB')
        self.rgb_button.setChecked(True)
        self.rgb_button.clicked.connect(self.update_feature_img)
        self.normalize_button = QCheckBox('Normalize')
        self.normalize_button.setChecked(True)
        self.normalize_button.clicked.connect(self.update_feature_img)
        self.channel_selector = QSpinBox()
        self.channel_selector_label = QLabel('Channel')
        self.channel_selector.valueChanged.connect(self.update_feature_img)

        # Reset bbox button
        self.reset_bbox_button = QPushButton("Reset BBox")
        self.reset_bbox_button.clicked.connect(self.image_in_display.reset_crop_bbox)

        # Help text
        self.help_text_display = QLabel()
        #help_text = open(str(Path(__file__).parent /'../../assets/help.txt'),'r').read()
        help_text = open('assets/help.txt','r').read()
        self.help_text_display.setText(help_text)

        self.feature_maps_table =  SelectionTableWidget()
        self.feature_maps_table.table.itemClicked.connect(self.update_feature_img)
        self.column1.addLayout(self._create_layout([self.image_path_label, self.image_path_input, self.load_button]))
        self.column1.addLayout(self._create_layout([self.normalize_button,self.rgb_button,self.channel_selector,
                                                    self.channel_selector_label]))
        self.column1.addWidget(self.reset_bbox_button,0)
        self.column1.addWidget(self.help_text_display)
        self.column1.addWidget(self.feature_maps_table,0)



        # Adding columns to main layout
        main_layout.addLayout(self.column1, 0)
        main_layout.addLayout(self.column2, 1)
        main_layout.addLayout(self.column3, 1)  # Third column is larger

        self.setCentralWidget(main_widget)


    def update_feature_img(self):
        # Get current selection
        row = self.feature_maps_table.table.currentRow()
        feature_map =self._feature_maps[self.feature_maps_table.table.item(row,0).text()]
        img_size    = self.image_out_display.size()
        feature_img = feature_map.get_numpy_img([feature_map.map.shape[-1],feature_map.map.shape[-2]])

        # Set channel selector
        self.channel_selector.setRange(0, feature_map.channels)
        self.channel_selector_label.setText(f'/{feature_map.channels} Channels')
        channel_idx = self.channel_selector.value()
        if self.rgb_button.isChecked() and channel_idx+2 < feature_map.channels:
            feature_img = feature_img[:,:,channel_idx:channel_idx+3]
        else:
            feature_img = feature_img[:,:,channel_idx, None].repeat(3, axis=2)

        if self.normalize_button.isChecked():
            feature_img = (feature_img - feature_img.min())/(feature_img.max() - feature_img.min())

        # Process feature image
        self._img_feat_display = feature_img
        self.display_images()


    def load(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if Path(file_path).suffix in ['.png', '.jpg', '.jpeg']:
            self.image_path_input.setText(file_path)
            self._input_img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

            self._img_in_display = self._input_img
            self._img_out_display = self._input_img
            self._img_feat_display = self._input_img

            self.network_inference(self._input_img)
            self.display_images()

    def display_images(self):
        self.image_in_display.set_image(self._img_in_display.copy())
        self.image_out_display.set_image(self._img_feat_display.copy())


    def network_inference(self, img: RGB255):
        if self.image_in_display.crop_bbox is not None:
            x1, y1, x2, y2 = self.image_in_display.crop_bbox
            img = img[y1:y2, x1:x2]
        img_tensor = np2torch(img)
        self._result     = self._network(img_tensor)
        self._feature_maps = get_feature_maps(self._result)

    def update_input(self):
        self.network_inference(self._input_img)
        self.display_images()
        self.update_feature_img()

    def _create_layout(self, widgets: list, type:str ='H',): # Choice of H, V, B
        layout = LAYOUTS[type]()
        if isinstance(widgets[0], QWidget):
            for widget in widgets:
                layout.addWidget(widget)
        elif isinstance(widgets[0], tuple):
            for widget, stretch in widgets:
                layout.addWidget(widget, stretch)
        return layout


class NetworkVisualizer:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.viz = NetworkVisualizerApplication()

    def register_module(self, module, depth=1, name=None):
        self.viz.register_module(module, depth, name)

    def set_forward(self, forward_func):
        self.viz.set_forward(forward_func)

    def run(self):
        self.viz.show()
        sys.exit(self.app.exec_())

