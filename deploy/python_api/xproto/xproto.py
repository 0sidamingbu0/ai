import sys
import os
import json
import time
import subprocess
sys.path.append("..")
sys.setdlopenflags(os.RTLD_LAZY)

import xstream   # noqa
import vision_type as vt    # noqa
from .global_var import GlobalVar   # noqa
from native_xproto import XPlgAsync, NativeVioPlg, XPlgAsyncHelper   # noqa
from native_xproto import SmartHelper   # noqa

__all__ = [
  "VioPlugin",
  "SmartPlugin",
]

global_variable = GlobalVar()


class XPluginAsync:
    """
    intermediate class
    native_xplgasync_: instance of native xpluginasync
    """

    def __init__(self, num=0):
        print("xpluginasync init")
        self.native_xplgasync_ = XPlgAsync() if num == 0 else XPlgAsync(num)
        self._xplgasync_helper_ = XPlgAsyncHelper()

    def reg_msg(self, msg_type, func):
        # check type
        self._xplgasync_helper_.register_msg(msg_type, func)

    def init(self):
        self.native_xplgasync_.init()

    def deinit(self):
        self.native_xplgasync_.deinit()

    def push_msg(self, msg):
        self.native_xplgasync_.push_msg(msg)


class VioPlugin(object):
    """
    VioPlugin
    currently not support multi_vio
    platform_: Which platform will the code run on. Support x3dev
    sensor_: sensor type. Support imx327.
    vio_type_: vio type. Support single_cam
    data_source_: hb data source. Support cache, jpg, nv12.
    cfg_file_: Path to vio config file.
    _native_vio_: An instance of native vioplugin.
    """

    def __gen_cfg_file(self):
        # load vio config file
        template_path = global_variable.get_vio_cfg_template_path()
        if global_variable.get_platform_prefix() == "x3":
            if global_variable.vio_type_ == "single_cam":
                template_path = template_path + ".cam"
            elif global_variable.vio_type_ == "single_feedback":
                template_path = template_path + ".fb"
            elif global_variable.vio_type_ == "multi_cam_sync":
                template_path = template_path + "multi_cam_sync"
            elif global_variable.vio_type_ == "multi_cam_async":
                template_path = template_path + "multi_cam_async"
            elif global_variable.vio_type_ == "fb_cam":
                template_path = template_path + "fb_cam"
        else:   # x2
            pass

        template_file = open(template_path)
        vio_cfg_json = json.load(template_file)
        template_file.close()

        # load panel camera system software vio config file
        panel_camera_vio_cfg_path = \
            vio_cfg_json["config_data"][0]["vio_cfg_file"]["panel_camera"]
        panel_camera_vio_cfg_json = json.load(open(panel_camera_vio_cfg_path))

        # modify data field in mono vio config file
        if not global_variable.sensor_ == "hg":
            if global_variable.platform_ == "x3dev":
                # TODO: config other sensors besides imx327
                if global_variable.sensor_ == "imx327":
                    panel_camera_vio_cfg_json["vio_vps_mode"] = 0
                    panel_camera_vio_cfg_json["sensor_port"] = 0
                    panel_camera_vio_cfg_json["i2c_bus"] = 5
                    panel_camera_vio_cfg_json["host_index"] = 1
            else:
                pass

        if panel_camera_vio_cfg_json is not None:
            cfg_path = "configs/py_vio_mono_cfg.json"
            out_file = open(cfg_path, "w")
            json.dump(panel_camera_vio_cfg_json, out_file, indent=4)
            out_file.close()
            vio_cfg_json["config_data"][0]["vio_cfg_file"]["panel_camera"] = \
                cfg_path

        # modify data field in vio config file
        if global_variable.sensor_ == "hg":
            if global_variable.data_source_ == "cache":
                vio_cfg_json["config_data"][0]["data_source"] = \
                    "cached_image_list"
            elif global_variable.data_source_ == "jpg":
                vio_cfg_json["config_data"][0]["data_source"] = \
                    "jpeg_image_list"
                vio_cfg_json["config_data"][0]["file_path"] = \
                    "configs/vio_hg/name.list"
            elif global_variable.data_source_ == "nv12":
                vio_cfg_json["config_data"][0]["data_source"] = \
                    "nv12_image_list"
                vio_cfg_json["config_data"][0]["file_path"] = \
                    "configs/vio_hg/name_nv12.list"
            else:
                print("Default data source is cached image list")
                vio_cfg_json["config_data"][0]["data_source"] = \
                    "cached_image_list"
        if global_variable.sensor_ == "usb_cam":
            # TODO(shiyu.fu): update cfg file path
            vio_cfg_json["data_source"] = "usb_cam"

        # save modified vio config file
        py_vio_cfg_path = "py_vio_cfg.json"
        out_file = open(py_vio_cfg_path, "w")
        json.dump(vio_cfg_json, out_file, indent=4)
        out_file.close()

        # return template_path
        return py_vio_cfg_path

    def __init__(
      self, platform, sensor="default", vio_type="single_cam",
      data_source="default", **kwargs):
        # check & set platform
        assert platform in global_variable.supported_platform_, \
            "Given platform is not supported"
        global_variable.set_platform(platform)
        # check & set sensor type
        assert sensor in global_variable.supported_sensor_, \
            "Given sensor type is not supported"
        global_variable.set_sensor(sensor)
        # check & set vio_type
        assert vio_type in global_variable.supported_vio_type_, \
            "Given vio type is not supported"
        global_variable.set_vio_type(vio_type)
        # check & set data source for hg
        assert data_source in GlobalVar.supported_data_source_, \
            "Given data_source is not supported"
        global_variable.set_data_source(data_source)

        self.cfg_file_ = self.__gen_cfg_file()
        self.msg_cb_ = {}
        self._native_vio_ = NativeVioPlg(self.cfg_file_)
        self._helper_ = SmartHelper()

    def start(self, sync_mode=False):
        self._native_vio_.set_mode(sync_mode)
        self._helper_.set_mode(sync_mode)
        if not self._native_vio_.is_inited():
            self._native_vio_.init()
        assert self._native_vio_.is_inited(), \
            "Cannot start vioplugin before init"
        ret = self._native_vio_.start()
        assert ret == 0, "Failed to start vioplugin, Code: %d" % ret

    def stop(self):
        ret = self._native_vio_.stop()
        assert ret == 0, "Failed to stop vioplugin, Code: %d" % ret
        ret = self._native_vio_.deinit()
        assert ret == 0, "Failed to deinit vioplugin, Code: %d" % ret

    def message_type(self):
        return ["XPLUGIN_IMAGE_MESSAGE", "XPLUGIN_DROP_MESSAGE"]

    def bind(self, msg_type, msg_cb=None):
        assert callable(msg_cb), "Callback is not callable for %s" % msg_type
        self.msg_cb_[msg_type] = msg_cb
        ret = self._native_vio_.add_msg_cb(msg_type, msg_cb)
        assert ret == 0, "Failed to add callback for %s" % msg_type

    def get_image(self):
        # sync but not real-time
        # need to monitor #images for feedback
        time.sleep(1)
        vio_msg = self._native_vio_.get_image()
        return self._helper_.to_xstream_data(vio_msg)

    def get_name_list(self):
        cfg_json = json.load(self.cfg_file_)
        return cfg_json["file_path"]


class SmartPlugin(XPluginAsync):
    """
    SmartPlugin
    xstream_sess_: xstream session
    workflow_: xstream workflow
    callback_: smart data callback
    serialize_: serializing function of custom message
    push_result_: push data to xplgflow, otherwise display in command line
    helper_: instance of SmartPlgHelper, C++/Python data conversion
    """

    def __init__(self, workflow, callback, serialize, push_result=False):
        XPluginAsync.__init__(self)
        self._xstream_sess_ = xstream.Session(workflow)
        self.callback_ = callback
        self.msg_cb_ = {}
        self.workflow_ = workflow
        self.push_result_ = push_result
        self._helper_ = SmartHelper()

    def start(self):
        # actual xstream callback
        def OnSmartData(*smart_rets):
            native_msg = self._helper_.to_native_msg(*smart_rets)
            if self.push_result_:
                self.push_msg(native_msg)
            else:
                self.callback_(*smart_rets)
        self._xstream_sess_.callback(OnSmartData)

        # flow message listener
        def OnFlowMsg(message):
            xstream_inputs = self._helper_.to_xstream_data(message)
            # get input names in workflow
            arg_names = self.workflow_.__code__.co_varnames
            arg_cnt = self.workflow_.__code__.co_argcount
            arg_names = arg_names[:arg_cnt]
            assert len(arg_names) == len(xstream_inputs), \
                "Input amounts do not match"
            # TODO(shiyu.fu): input name mismatch
            for idx, arg_name in enumerate(arg_names):
                assert arg_name in xstream_inputs, \
                    "No input with name %s" % arg_name

            self._xstream_sess_.forward(**xstream_inputs)

        for msg_type, msg_cb in self.msg_cb_.items():
            if msg_type == "XPLUGIN_IMAGE_MESSAGE":
                super().reg_msg(msg_type, OnFlowMsg)
            else:
                assert callable(msg_cb), \
                    "Callback for %s is not callable" % msg_type
                super().reg_msg(msg_type, msg_cb)
        super().init()

    def stop(self):
        self._xstream_sess_.close()

    def message_type(self):
        return ["XPLUGIN_SMART_MESSAGE"]

    def bind(self, msg_type, msg_cb=None):
        self.msg_cb_[msg_type] = msg_cb

    def feed(self, inputs):
        # get input names in workflow
        arg_names = self.workflow_.__code__.co_varnames
        arg_cnt = self.workflow_.__code__.co_argcount
        arg_names = arg_names[:arg_cnt]
        assert len(arg_names) == len(inputs), \
            "Input amounts do not match"
        # TODO(shiyu.fu): input name mismatch
        for idx, arg_name in enumerate(arg_names):
            assert arg_name in inputs, \
                "No input with name %s" % arg_name
        self._xstream_sess_.callback(self.callback_)
        # input data have already been converted
        # from VioMessage to BaseDataWrapper
        # TODO(shiyu.fu): using xstream syncpredict
        self._xstream_sess_.forward(**inputs)
