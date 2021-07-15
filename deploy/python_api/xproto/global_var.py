import os
import subprocess
from .commands import *


class GlobalVar:
    supported_platform_ = {"96board": "configs/vio_config.json.96board",
                           "2610": "configs/vio_config.json.2610",
                           "x3dev": "configs/vio_config.json.x3dev",
                           "x3svb": "configs/vio_config.json.x3dev"}
    supported_sensor_ = {
        "default": "", "hg": "hg", "usb_cam": "hg",
        "imx327": "configs/vio/x3dev/iot_vio_x3_imx327.json",
        "os8a10": "configs/vio/x3dev/iot_vio_x3_os8a10.json",
        "s5kgm": "configs/vio/x3dev/iot_vio_x3_s5kgm1sp.json",
        "s5kgm_2160p": "configs/vio/x3dev/iot_vio_x3_s5kgm1sp_2160p.json"}
    supported_vio_type_ = ["single_cam", "single_feedback", "multi_cam_sync",
                           "multi_cam_async", "single_cam_single_feedback"]
    supported_data_source_ = ["default", "cache", "jpg", "nv12"]
    layer_ = {"96board": 4, "2610": 4, "hg": 0, "usb_cam": 0,
              "x3dev": {
                  "imx327": 0, "os8a10": 4, "os8a10_1080p": 0,
                  "s5kgm": 5, "s5kgm_2160p": 4, "hg": 0, "usb_cam": 0
              }}
    image_width_ = {"96board": 960, "2610": 540, "hg": 1920, "usb_cam": 1920,
                    "x3dev": {
                        "imx327": 1920, "os8a10": 1920, "os8a10_1080p": 1920,
                        "s5kgm": 1024, "s5kgm_2160p": 1920,
                        "hg": 1920, "usb_cam": 1920
                    }}
    image_height_ = {"96board": 960, "2610": 540, "hg": 1080, "usb_cam": 1080,
                     "x3dev": {
                         "imx327": 1080, "os8a10": 1080, "os8a10_1080p": 1080,
                         "s5kgm": 768, "s5kgm_2160p": 1080,
                         "hg": 1080, "usb_cam": 1080
                     }}
    layer_2160p_ = {"default": None, "s5kgm": None, "imx327": -1,
                    "os8a10": 0, "os8a10_1080p": -1, "s5kgm_2160p": 0,
                    "hg": -1}
    layer_1080p_ = {"default": None, "s5kgm": None, "imx327": 0,
                    "os8a10": 4, "os8a10_1080p": 0, "s5kgm_2160p": 4,
                    "hg": 0}
    layer_720p_ = {"default": None, "s5kgm": None, "imx327": 1,
                   "os8a10": 5, "os8a10_1080p": 1, "s5kgm_2160p": 5,
                   "hg": 1}

    def __replace_method_cfg_file(self, common_cmd):
        cmd = ""
        if self.sensor_ == "imx327":
            cmd = common_cmd + command_imx327   # TODO: currently for x3 only
        elif self.sensor_ == "os8a10":
            cmd = command_os8a10
        elif self.sensor_ == "os8a10_1080p":
            cmd = command_os8a10_1080p
        elif self.sensor_ == "s5kgm":
            cmd = command_s5kgm
        elif self.sensor_ == "s5kgm_2160p":
            cmd = command_s5kgm_2160p
        elif self.sensor_ == "hg":
            cmd = command_hg
        elif self.sensor_ == "usb_cam":
            cmd = command_usb_cam

        subprocess.Popen(cmd, shell=True)   # TODO: display output & check

    def set_platform(self, platform):
        self.platform_ = platform
        if platform == "x3dev" or platform == "x3svb":
            subprocess.Popen(command_x3_common, shell=True)

    def set_sensor(self, sensor):
        self.sensor_ = sensor
        if self.platform_ == "x3dev" or self.platform_ == "x3svb":
            self.__replace_method_cfg_file(command_x3)

    def set_vio_type(self, vio_type):
        self.vio_type_ = vio_type

    def set_data_source(self, data_source):
        self.data_source_ = data_source

    def get_platform_prefix(self):
        return self.platform_[:2]

    def get_vio_cfg_template_path(self):
        return GlobalVar.supported_platform_[self.platform_]

    def get_mono_vio_cfg_path(self):
        return GlobalVar.supported_sensor_[self.sensor_]

    def get_layer(self):
        layer = GlobalVar.layer_[self.platform_]
        if type(layer) is not int:
            layer = layer[self.sensor_]
        assert type(layer) is int, "wrong layer dtype"
        return layer

    def get_image_width(self):
        width = GlobalVar.image_width_[self.platform_]
        if type(width) is not int:
            width = width[self.sensor_]
        assert type(width) is int, "wrong image_width dtype"
        return width

    def get_image_height(self):
        height = GlobalVar.image_height_[self.platform_]
        if type(height) is not int:
            height = height[self.sensor_]
        assert type(height) is int, "wrong image_height dtype"
        return height

    def get_2160p_layer(self):
        return GlobalVar.layer_2160p_[self.sensor_]

    def get_1080p_layer(self):
        return GlobalVar.layer_1080p_[self.sensor_]

    def get_720p_layer(self):
        return GlobalVar.layer_720p_[self.sensor_]
