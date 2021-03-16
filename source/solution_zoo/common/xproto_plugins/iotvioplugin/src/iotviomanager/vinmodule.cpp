/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author: yong.wu
 * @Mail: yong.wu@horizon.ai
 * @Date: 2020-08-12
 * @Version: v1.0.0
 * @Brief: IOT VIN Module for Horizon VIO System.
 */
#include <dlfcn.h>
#include <string.h>
#include "iotviomanager/vinmodule.h"
#include "iotviomanager/violog.h"
#include "iotviomanager/vinparams.h"
#include "hobotlog/hobotlog.hpp"

namespace horizon {
namespace vision {
namespace xproto {
namespace vioplugin {

int VinModule::Init() {
  int ret = -1;

  ret = VinInit();
  if (ret) {
    LOGE << "VinModule VinInit failed!";
    return ret;
  }

  ret = CamInit();
  if (ret) {
    LOGE << "VinModule CamInit failed!";
    return ret;
  }

  return 0;
}

int VinModule::DeInit() {
  int ret;

  ret = VinDeInit();
  if (ret) {
    LOGE << "VinModule VinDeInit failed!";
    return ret;
  }

  ret = CamDeInit();
  if (ret) {
    LOGE << "VinModule CamDeInit failed!";
    return ret;
  }

  return 0;
}

int VinModule::Start() {
  int ret;

  LOGD << "Enter vinmodule start, pipe_id: " << pipe_id_;
  ret = VinStart();
  if (ret) {
    LOGE << "VinModule VinStart failed, pipe_id: " << pipe_id_;
    return ret;
  }

  ret = CamStart();
  if (ret) {
    LOGE << "VinModule CamStart failed, pipe_id: " << pipe_id_;
    return ret;
  }

  return 0;
}

int VinModule::Stop() {
  int ret;

  LOGD << "Enter vinmodule stop, pipe_id: " << pipe_id_;
  ret = CamStop();
  if (ret) {
    LOGE << "VinModule CamStop failed, pipe_id: " << pipe_id_;
    return ret;
  }

  ret = VinStop();
  if (ret) {
    LOGE << "VinModule VinStop failed, pipe_id: " << pipe_id_;
    return ret;
  }

  return 0;
}

int VinModule::HbSensorPlgInit() {
  std::string sensor_path = vio_cfg_.vin_cfg.sensor_info.sensor_path;
  std::string sensor_name = vio_cfg_.vin_cfg.sensor_info.sensor_name;
  int sensor_type = vio_cfg_.vin_cfg.sensor_info.sensor_type;
  int need_clk = vio_cfg_.vin_cfg.sensor_info.need_clk;

  if (sensor_name.empty() || sensor_path.empty()) {
    LOGE << "sensor plugin name or path is nullptr";
    return -1;
  }
  std::string name_buff = sensor_path + "lib" + sensor_name + ".so";
  const char *c_name_buff = name_buff.c_str();
  sensor_plg_cfg_.sensor_fd = dlopen(c_name_buff, RTLD_LAZY);
  if (sensor_plg_cfg_.sensor_fd == nullptr) {
    LOGE << "dlopen plugin sensor: " << name_buff << " error: "
      << dlerror();
    return -1;
  }
  sensor_plg_cfg_.sensor_ops =
    dlsym(sensor_plg_cfg_.sensor_fd, sensor_name.c_str());
  if (sensor_plg_cfg_.sensor_ops == nullptr) {
    LOGE << "dlsym plugin sensor: " << name_buff << " error: "
      << dlerror();
    return -1;
  }
  sensor_plg_cfg_.sensor_name = sensor_name.c_str();
  sensor_plg_cfg_.sensor_type = sensor_type;
  sensor_plg_cfg_.need_clk = need_clk;

  return 0;
}

int VinModule::HbSensorPlgDeInit() {
  if (sensor_plg_cfg_.sensor_fd) {
    dlclose(sensor_plg_cfg_.sensor_fd);
  }
  return 0;
}

int VinModule::CamInit() {
  int ret = -1;
  int pipe_id = pipe_id_;
  int sensor_id = vio_cfg_.vin_cfg.sensor_info.sensor_id;
  int sensor_port = vio_cfg_.vin_cfg.sensor_info.sensor_port;
  int i2c_bus = vio_cfg_.vin_cfg.sensor_info.i2c_bus;
  int need_clk = vio_cfg_.vin_cfg.sensor_info.need_clk;
  int serdes_idx = vio_cfg_.vin_cfg.sensor_info.serdes_index;
  int serdes_port = vio_cfg_.vin_cfg.sensor_info.serdes_port;
  int mipi_idx = vio_cfg_.vin_cfg.mipi_info.mipi_index;

  if (need_clk == 1) {
    ret = HbEnableSensorClk(mipi_idx);
    if (ret) {
    LOGE << "hb enable sensor clk, ret: " << ret
      << " pipe_id: " << pipe_id
      << " sensor_id: " << sensor_id
      << " mipi_idx: " << mipi_idx;
      return ret;
    }
  }
  ret = HbSensorInit(pipe_id, sensor_id, i2c_bus, sensor_port,
      mipi_idx, serdes_idx, serdes_port);
  if (ret < 0) {
    LOGE << "hb sensor init error, ret: " << ret
      << " pipe_id: " << pipe_id
      << " sensor_id: " << sensor_id
      << " mipi_idx: " << mipi_idx;
    return ret;
  }
  ret = HbMipiInit(sensor_id, mipi_idx);
  if (ret < 0) {
    LOGE << "hb mipi init error, ret: " << ret
      << " pipe_id: " << pipe_id
      << " sensor_id: " << sensor_id
      << " mipi_idx: " << mipi_idx;
    return ret;
  }
  cam_init_flag_ = true;

  return 0;
}

int VinModule::CamDeInit() {
  int ret = -1;
  int pipe_id = pipe_id_;
  int need_clk = vio_cfg_.vin_cfg.sensor_info.need_clk;
  int mipi_idx = vio_cfg_.vin_cfg.mipi_info.mipi_index;

  ret = HbMipiDeInit(mipi_idx);
  if (ret < 0) {
    LOGE << "hb mipi deinit error, ret: " << ret
      << " pipe_id: " << pipe_id << " mipi_idx: " << mipi_idx;
    return ret;
  }
  ret = HbSensorDeInit(pipe_id);
  if (ret < 0) {
    LOGE << "hb sensor deinit error, ret: " << ret
      << " pipe_id: " << pipe_id << " mipi_idx: " << mipi_idx;
    return ret;
  }
  if (need_clk == 1) {
    ret = HbDisableSensorClk(mipi_idx);
    if (ret) {
    LOGE << "hb disable sensor clock error, ret: " << ret
      << " pipe_id: " << pipe_id << " mipi_idx: " << mipi_idx;
      return ret;
    }
  }

  return 0;
}

int VinModule::CamStart() {
  int ret = -1;
  int pipe_id = pipe_id_;
  int mipi_idx = vio_cfg_.vin_cfg.mipi_info.mipi_index;

  LOGD << "Enter cam start, pipe_id: " << pipe_id
    << " mipi_idx: " << mipi_idx;
  ret = HbSensorStart(pipe_id);
  if (ret < 0) {
    LOGE << "hb sensor start error, ret: " << ret
      << " pipe_id: " << pipe_id << " mipi_idx: " << mipi_idx;
    return ret;
  }

  ret = HbMipiStart(mipi_idx);
  if (ret < 0) {
    LOGE << "hb mipi start error, ret: " << ret
      << " pipe_id: " << pipe_id << " mipi_idx: " << mipi_idx;
    return ret;
  }

  return 0;
}

int VinModule::CamStop() {
  int ret = -1;
  int pipe_id = pipe_id_;
  int mipi_idx = vio_cfg_.vin_cfg.mipi_info.mipi_index;

  LOGD << "Enter sensor stop, pipe_id: " << pipe_id;
  ret = HbSensorStop(pipe_id);
  if (ret < 0) {
    LOGE << "hb sensor stop error, ret: " << ret
      << " pipe_id: " << pipe_id << " mipi_idx: " << mipi_idx;
    return ret;
  }
  LOGD << "Enter mipi stop, mipi_idx: " << mipi_idx;
  ret = HbMipiStop(mipi_idx);
  if (ret < 0) {
    LOGE << "hb mipi stop error, ret: " << ret
      << " pipe_id: " << pipe_id << " mipi_idx: " << mipi_idx;
    return ret;
  }

  return 0;
}

int VinModule::HbSensorInit(int dev_id, int sensor_id, int bus, int port,
    int mipi_idx, int sedres_index, int sedres_port) {
  int ret = -1;
  int extra_mode = vio_cfg_.vin_cfg.sensor_info.extra_mode;
  MIPI_SENSOR_INFO_S *snsinfo = NULL;

  snsinfo = &cam_params_.sensor_info;
  memset(snsinfo, 0, sizeof(MIPI_SENSOR_INFO_S));
  if (sensor_plg_en_ == true) {
    SnsPlgModule *plg_handle =
      reinterpret_cast<SnsPlgModule *>(sensor_plg_cfg_.sensor_ops);
    ret = plg_handle->get_sns_attr(&sensor_plg_cfg_, snsinfo);
    if (ret) {
      LOGE << "sensor plugin get sensor attr falied, ret: " << ret;
      return ret;
    }
  } else {
    HbMipiGetSnsAttrBySns(static_cast<MipiSensorType>(sensor_id),
        snsinfo);
  }

  if (sensor_id == kOV10635_30FPS_720p_960_YUV_LINE_CONCATENATED) {
    HB_MIPI_SetExtraMode(snsinfo, extra_mode);
  }
  HB_MIPI_SetBus(snsinfo, bus);
  HB_MIPI_SetPort(snsinfo, port);
  HB_MIPI_SensorBindSerdes(snsinfo, sedres_index, sedres_port);
  HB_MIPI_SensorBindMipi(snsinfo, mipi_idx);
  snsinfo->sensorInfo.gpio_num = vio_cfg_.vin_cfg.sensor_info.gpio_num;
  snsinfo->sensorInfo.gpio_pin[0] = vio_cfg_.vin_cfg.sensor_info.gpio_pin;
  snsinfo->sensorInfo.gpio_level[0] = vio_cfg_.vin_cfg.sensor_info.gpio_level;

  HbPrintSensorInfo(snsinfo);
  ret = HB_MIPI_InitSensor(dev_id, snsinfo);
  if (ret < 0) {
    LOGE << "hb mipi init sensor error, ret: " << ret
      << " dev_id: " << dev_id;
    return ret;
  }
  LOGD << "hb sensor init success...";

  return 0;
}

int VinModule::HbSensorDeInit(int dev_id) {
  int ret = -1;

  ret = HB_MIPI_DeinitSensor(dev_id);
  if (ret < 0) {
    LOGE << "hb mipi deinit sensor error, ret: " << ret
      << " dev_id: " << dev_id;
    return ret;
  }

  return 0;
}

int VinModule::HbSensorStart(int dev_id) {
  int ret = -1;

  ret = HB_MIPI_ResetSensor(dev_id);
  if (ret < 0) {
    LOGE << "hb mipi reset sensor, ret: " << ret
      << " dev_id: " << dev_id;
    return ret;
  }

  return 0;
}

int VinModule::HbSensorStop(int dev_id) {
  int ret = -1;

  ret = HB_MIPI_UnresetSensor(dev_id);
  if (ret < 0) {
    LOGE << "hb mipi unreset sensor, ret: " << ret
      << " dev_id: " << dev_id;
    return ret;
  }

  return 0;
}

int VinModule::HbEnableSensorClk(int mipi_idx) {
  int ret = -1;

  ret = HB_MIPI_EnableSensorClock(mipi_idx);
  if (ret < 0) {
    LOGE << "hb mipi enable sensor error, ret: " << ret
      << " mipi_idx: " << mipi_idx;
    return ret;
  }

  return 0;
}

int VinModule::HbDisableSensorClk(int mipi_idx) {
  int ret = -1;

  ret = HB_MIPI_DisableSensorClock(mipi_idx);
  if (ret < 0) {
    LOGE << "hb mipi disable sensor error, ret: " << ret
      << " mipi_idx: " << mipi_idx;
    return ret;
  }

  return 0;
}

int VinModule::HbMipiInit(int sensor_id, int mipi_idx) {
  int ret = -1;
  MIPI_ATTR_S *mipi_attr = NULL;

  mipi_attr = &cam_params_.mipi_info;
  memset(mipi_attr, 0, sizeof(MIPI_ATTR_S));

  if (sensor_plg_en_ == true) {
    SnsPlgModule *plg_handle =
      reinterpret_cast<SnsPlgModule *>(sensor_plg_cfg_.sensor_ops);
    ret = plg_handle->get_mipi_attr(&sensor_plg_cfg_, mipi_attr);
    if (ret) {
      LOGE << "sensor plugin get mipi attr falied, ret: " << ret;
      return ret;
    }
  } else {
    HbMipiGetMipiAttrBySns(static_cast<MipiSensorType>(sensor_id),
        mipi_attr);
  }

  ret = HB_MIPI_SetMipiAttr(mipi_idx, mipi_attr);
  if (ret < 0) {
    LOGE << "hb mipi set mipi attr error, ret: " << ret
      << " sensor_id:" << sensor_id << " mipi_idx:"  << mipi_idx;
    return ret;
  }
  LOGD << "hb mipi init success...";

  return 0;
}

int VinModule::HbMipiDeInit(int mipi_idx) {
  int ret = -1;

  ret = HB_MIPI_Clear(mipi_idx);
  if (ret < 0) {
    LOGE << "hb mipi clear error, ret: " << ret
      << " mipi_idx: " << mipi_idx;
    return ret;
  }
  LOGD << "hb mipi deinit success...";

  return 0;
}

int VinModule::HbMipiStart(int mipi_idx) {
  int ret = -1;

  ret = HB_MIPI_ResetMipi(mipi_idx);
  if (ret < 0) {
    LOGE << "hb mipi reset mipi error, ret: " << ret
      << " mipi_idx: " << mipi_idx;
    return ret;
  }

  return 0;
}

int VinModule::HbMipiStop(int mipi_idx) {
  int ret = -1;
  ret = HB_MIPI_UnresetMipi(mipi_idx);
  if (ret < 0) {
    LOGE << "HB_MIPI_UnresetMipi error, ret: " << ret
      << " mipi_idx: " << mipi_idx;
    return ret;
  }

  return 0;
}

int VinModule::VinStart() {
  int ret = -1;
  int pipe_id = pipe_id_;
  LOGD << "Enter vinmodule start, pipe_id:" << pipe_id;

  ret = HB_VIN_EnableChn(pipe_id, 0);  // dwe start
  if (ret < 0) {
    LOGE << "HB_VIN_EnableChn error, ret: " << ret
      << " pipe_id: " << pipe_id;
    return ret;
  }

  ret = HB_VIN_StartPipe(pipe_id);  // isp start
  if (ret < 0) {
    LOGE << "HB_VIN_StartPipe error, ret: " << ret
      << " pipe_id: " << pipe_id;
    return ret;
  }
  ret = HB_VIN_EnableDev(pipe_id);  // sif start && start thread
  if (ret < 0) {
    LOGE << "HB_VIN_EnableDev error, ret: " << ret
      << " pipe_id: " << pipe_id;
    return ret;
  }

  return 0;
}

int VinModule::VinStop() {
  int pipe_id = pipe_id_;

  LOGI << "Enter vinmodule stop, pipe_id: " << pipe_id;
  HB_VIN_DisableDev(pipe_id);     // thread stop && sif stop
  HB_VIN_StopPipe(pipe_id);       // isp stop
  HB_VIN_DisableChn(pipe_id, 1);  // dwe stop

  return 0;
}

int VinModule::VinDeInit() {
  int pipe_id = pipe_id_;

  HB_VIN_DestroyDev(pipe_id);     // sif deinit && destroy
  HB_VIN_DestroyChn(pipe_id, 1);  // dwe deinit
  HB_VIN_DestroyPipe(pipe_id);    // isp deinit && destroy
  if (vio_cfg_.vin_cfg.vin_fd_en) {
    HB_VIN_CloseFd();
  }

  if (sensor_plg_en_ == true) {
    HbSensorPlgDeInit();
    sensor_plg_en_ = false;
  }
  return 0;
}

void VinModule::HbDisCropSet(uint32_t pipe_id, uint32_t event,
    VIN_DIS_MV_INFO_S *data, void *userdata) {
  LOGD << "data gmvX: "    << data->gmvX;
  LOGD << "data gmvY: "    << data->gmvY;
  LOGD << "data xUpdate: " << data->xUpdate;
  LOGD << "data yUpdate: " << data->yUpdate;

  return;
}

int VinModule::VinInit() {
  int ret = -1;
  int vin_vps_mode = vio_cfg_.vin_vps_mode;
  int sensor_id = vio_cfg_.vin_cfg.sensor_info.sensor_id;
  int mipi_idx = vio_cfg_.vin_cfg.mipi_info.mipi_index;
  int vc_idx = vio_cfg_.vin_cfg.mipi_info.vc_index;
  int dol2_vc_idx = vio_cfg_.vin_cfg.mipi_info.dol2_vc_index;
  int pipe_id = pipe_id_;
  int plugin_en = vio_cfg_.vin_cfg.sensor_info.plugin_en;

  LOGI << "Enter VinInit, plugin_en: " << plugin_en;
  if (plugin_en == 1) {
    ret = HbSensorPlgInit();
    if (ret) {
      LOGE << "VinModule Sensor plugin init failed!";
      return ret;
    }
    sensor_plg_en_ = true;
  }

  VIN_DEV_ATTR_S *devinfo = NULL;
  VIN_PIPE_ATTR_S *pipeinfo = NULL;
  VIN_DIS_ATTR_S *disinfo = NULL;
  VIN_LDC_ATTR_S *ldcinfo = NULL;
  VIN_DEV_ATTR_EX_S *devexinfo = NULL;
  VIN_DIS_CALLBACK_S pstDISCallback;
  pstDISCallback.VIN_DIS_DATA_CB = HbDisCropSet;

  LOGD << "Enter Vin init, pipe_id: " << pipe_id;
  devinfo = &vin_params_.dev_info;
  devexinfo = &vin_params_.devex_info;
  pipeinfo = &vin_params_.pipe_info;
  disinfo = &vin_params_.dis_info;
  ldcinfo = &vin_params_.ldc_info;

  memset(devinfo, 0, sizeof(VIN_DEV_ATTR_S));
  memset(devexinfo, 0, sizeof(VIN_DEV_ATTR_EX_S));
  memset(pipeinfo, 0, sizeof(VIN_PIPE_ATTR_S));
  memset(disinfo, 0, sizeof(VIN_DIS_ATTR_S));
  memset(ldcinfo, 0, sizeof(VIN_LDC_ATTR_S));

  /* get default vin params */
  if (sensor_plg_en_ == true) {
    SnsPlgModule *plg_handle =
      reinterpret_cast<SnsPlgModule *>(sensor_plg_cfg_.sensor_ops);
    ret = plg_handle->get_vin_dev_attr(&sensor_plg_cfg_, devinfo);
    if (ret) {
      LOGE << "sensor plugin get vin dev attr falied, ret: " << ret;
      return ret;
    }
    ret = plg_handle->get_vin_dev_attr_ex(&sensor_plg_cfg_, devexinfo);
    if (ret) {
      LOGE << "sensor plugin get vin dev ex attr falied, ret: " << ret;
      return ret;
    }
    ret = plg_handle->get_vin_pipe_attr(&sensor_plg_cfg_, pipeinfo);
    if (ret) {
      LOGE << "sensor plugin get vin pipe attr falied, ret: " << ret;
      return ret;
    }
    ret = plg_handle->get_dis_attr(&sensor_plg_cfg_, disinfo);
    if (ret) {
      LOGE << "sensor plugin get vin dis attr falied, ret: " << ret;
      return ret;
    }
    ret = plg_handle->get_ldc_attr(&sensor_plg_cfg_, ldcinfo);
    if (ret) {
      LOGE << "sensor plugin get vin ldc attr falied, ret: " << ret;
      return ret;
    }
  } else {
    HbVinGetDevAttrBySns(static_cast<MipiSensorType>(sensor_id), devinfo);
    HbVinGetDevAttrExBySns(static_cast<MipiSensorType>(sensor_id),
        devexinfo);
    HbVinGetPipeAttrBySns(static_cast<MipiSensorType>(sensor_id), pipeinfo);
    HbVinGetDisAttrBySns(static_cast<MipiSensorType>(sensor_id), disinfo);
    HbVinGetLdcAttrBySns(static_cast<MipiSensorType>(sensor_id), ldcinfo);
  }
  /* set vin params */
  HbVinSetConfig();
  /* print vin params */
  HbPrintSensorDevInfo(devinfo);
  HbPrintSensorPipeInfo(pipeinfo);

  ret = HB_SYS_SetVINVPSMode(pipe_id,
      static_cast<SYS_VIN_VPS_MODE_E>(vin_vps_mode));
  if (ret < 0) {
    LOGE << "HB_SYS_SetVINVPSMode error, ret: " << ret;
    return ret;
  }
  ret = HB_VIN_CreatePipe(pipe_id, pipeinfo);  // isp init
  if (ret < 0) {
    LOGE << "HB_VIN_CreatePipe error, ret: " << ret;
    return ret;
  }
  ret = HB_VIN_SetMipiBindDev(pipe_id, mipi_idx);
  if (ret < 0) {
    LOGE << "HB_VIN_SetMipiBindDev error, ret: " << ret;
    return ret;
  }
  ret = HB_VIN_SetDevVCNumber(pipe_id, vc_idx);
  if (ret < 0) {
    LOGE << "HB_VIN_SetDevVCNumber error, ret: " << ret;
    return ret;
  }
  if (sensor_id == kIMX327_30FPS_2228P_RAW12_DOL2 ||
      sensor_id == kOS8A10_30FPS_3840P_RAW10_DOL2) {
    ret = HB_VIN_AddDevVCNumber(pipe_id, dol2_vc_idx);
    if (ret < 0) {
      LOGE << "HB_VIN_AddDevVCNumber error, ret: " << ret;
      return ret;
    }
  }
  ret = HB_VIN_SetDevAttr(pipe_id, devinfo);  // sif init
  if (ret < 0) {
      LOGE << "HB_VIN_SetDevAttr error, ret: " << ret;
    return ret;
  }
  if (vio_cfg_.vin_cfg.sif_info.need_md) {
    ret = HB_VIN_SetDevAttrEx(pipe_id, devexinfo);
    if (ret < 0) {
      LOGE << "HB_VIN_SetDevAttrEx error, ret: " << ret;
      return ret;
    }
  }
  ret = HB_VIN_SetPipeAttr(pipe_id, pipeinfo);  // isp init
  if (ret < 0) {
      LOGE << "HB_VIN_SetPipeAttr error, ret: " << ret;
    goto pipe_err;
  }
  ret = HB_VIN_SetChnDISAttr(pipe_id, 1, disinfo);  //  dis init
  if (ret < 0) {
      LOGE << "HB_VIN_SetChnDISAttr error, ret: " << ret;
    goto pipe_err;
  }
  if (vio_cfg_.vin_cfg.dwe_info.dis_en) {
    HB_VIN_RegisterDisCallback(pipe_id, &pstDISCallback);
  }
  ret = HB_VIN_SetChnLDCAttr(pipe_id, 1, ldcinfo);  //  ldc init
  if (ret < 0) {
      LOGE << "HB_VIN_SetChnLDCAttr error, ret: " << ret;
    goto pipe_err;
  }
  ret = HB_VIN_SetChnAttr(pipe_id, 1);  //  dwe init
  if (ret < 0) {
      LOGE << "HB_VIN_SetChnAttr error, ret: " << ret;
    goto pipe_err;
  }
  ret = HB_VIN_SetDevBindPipe(pipe_id, pipe_id);  //  bind init
  if (ret < 0) {
      LOGE << "HB_VIN_SetDevBindPipe error, ret: " << ret;
    goto chn_err;
  }

  if (vio_cfg_.vin_cfg.vin_fd_en) {
    vin_fd_ = HB_VIN_GetChnFd(pipe_id, 0);
    if (vin_fd_ < 0) {
      LOGE << "HB_VIN_GetChnFd error, ret: " << ret;
    }
  }

  vin_init_flag_ = true;
  return 0;

chn_err:
  HB_VIN_DestroyPipe(pipe_id);  // isp && dwe deinit
pipe_err:
  HB_VIN_DestroyDev(pipe_id);   // sif deinit

  return ret;
}

int VinModule::GetVinConfig(IotVinParams *vin_cfg) {
  CHECK_PARAMS_VALID(vin_cfg);
  if (vin_init_flag_ == false) {
      LOGE << "vin has not init!";
    return -1;
  }
  *vin_cfg = vin_params_;

  return 0;
}

int VinModule::GetCamConfig(IotVinParams *cam_cfg) {
  CHECK_PARAMS_VALID(cam_cfg);
  if (cam_init_flag_ == false) {
      LOGE << "cam has not init!";
    return -1;
  }
  *cam_cfg = vin_params_;

  return 0;
}

int VinModule::HbMipiGetSnsAttrBySns(MipiSensorType sensor_type,
    MIPI_SENSOR_INFO_S *pst_sns_attr) {

  switch (sensor_type) {
    case kIMX327_30FPS_1952P_RAW12_LINEAR:
      memcpy(pst_sns_attr,
          &SENSOR_4LANE_IMX327_30FPS_12BIT_LINEAR_INFO,
          sizeof(MIPI_SENSOR_INFO_S));
      break;
    case kIMX327_30FPS_2228P_RAW12_DOL2:
      memcpy(pst_sns_attr,
          &SENSOR_4LANE_IMX327_30FPS_12BIT_DOL2_INFO,
          sizeof(MIPI_SENSOR_INFO_S));
      break;
    case kAR0233_30FPS_1080P_RAW12_954_PWL:
      memcpy(pst_sns_attr,
          &SENSOR_4LANE_AR0233_30FPS_12BIT_1080P_954_INFO,
          sizeof(MIPI_SENSOR_INFO_S));
      break;
    case kAR0233_30FPS_1080P_RAW12_960_PWL:
      memcpy(pst_sns_attr,
          &SENSOR_4LANE_AR0233_30FPS_12BIT_1080P_960_INFO,
          sizeof(MIPI_SENSOR_INFO_S));
      break;
    case kOS8A10_30FPS_3840P_RAW10_LINEAR:
      memcpy(pst_sns_attr,
          &SENSOR_OS8A10_30FPS_10BIT_LINEAR_INFO,
          sizeof(MIPI_SENSOR_INFO_S));
      break;
    case kOS8A10_30FPS_3840P_RAW10_DOL2:
      memcpy(pst_sns_attr,
          &SENSOR_OS8A10_30FPS_10BIT_DOL2_INFO,
          sizeof(MIPI_SENSOR_INFO_S));
      break;
    case kOV10635_30FPS_720p_954_YUV:
      memcpy(pst_sns_attr,
          &SENSOR_2LANE_OV10635_30FPS_YUV_720P_954_INFO,
          sizeof(MIPI_SENSOR_INFO_S));
      break;
    case kOV10635_30FPS_720p_960_YUV:
    case kOV10635_30FPS_720p_960_YUV_LINE_CONCATENATED:
      memcpy(pst_sns_attr,
          &SENSOR_2LANE_OV10635_30FPS_YUV_720P_960_INFO,
          sizeof(MIPI_SENSOR_INFO_S));
      break;
    case kSIF_TEST_PATTERN_1080P:
    case kSIF_TEST_PATTERN_4K:
    case kSIF_TEST_PATTERN_YUV_720P:
    case kSIF_TEST_PATTERN_12M_RAW12:
      memcpy(pst_sns_attr,
          &SENSOR_TESTPATTERN_INFO,
          sizeof(MIPI_SENSOR_INFO_S));
      break;
    case kS5KGM1SP_30FPS_4000x3000_RAW10:
      memcpy(pst_sns_attr,
          &SENSOR_S5KGM1SP_30FPS_10BIT_LINEAR_INFO,
          sizeof(MIPI_SENSOR_INFO_S));
      break;
    case kF37_30FPS_1080P_RAW10_LINEAR:
      memcpy(pst_sns_attr,
          &SENSOR_F37_30FPS_10BIT_LINEAR_INFO,
          sizeof(MIPI_SENSOR_INFO_S));
      break;
    case kSC8238_30FPS_3840P_RAW10_LINEAR:
      memcpy(pst_sns_attr,
             &SENSOR_SC8238_30FPS_10BIT_LINEAR_INFO,
             sizeof(MIPI_SENSOR_INFO_S));
      break;
    case kSC8238_30FPS_3840P_RAW10_DOL2:
      memcpy(pst_sns_attr,
             &SENSOR_SC8238_30FPS_10BIT_DOL2_INFO,
             sizeof(MIPI_SENSOR_INFO_S));
      break;
    case kGC4C33_30FPS_1440P_LINEAR:
      memcpy(pst_sns_attr,
             &SENSOR_GC4C33_30FPS_1440P_LINEAR_INFO,
             sizeof(MIPI_SENSOR_INFO_S));
      break;
    default:
      LOGE << "not surpport sensor type sensor_type" << sensor_type;
      break;
  }
  LOGD << "Get sensor attr success...";
  return 0;
}

int VinModule::HbMipiGetMipiAttrBySns(MipiSensorType sensor_type,
    MIPI_ATTR_S *pst_mipi_attr) {
  int need_clk;

  need_clk = vio_cfg_.vin_cfg.sensor_info.need_clk;
  switch (sensor_type) {
    case kIMX327_30FPS_1952P_RAW12_LINEAR:
      memcpy(pst_mipi_attr,
          &MIPI_4LANE_SENSOR_IMX327_30FPS_12BIT_NORMAL_ATTR,
          sizeof(MIPI_ATTR_S));
      break;
    case kIMX327_30FPS_2228P_RAW12_DOL2:
      memcpy(pst_mipi_attr,
          &MIPI_4LANE_SENSOR_IMX327_30FPS_12BIT_DOL2_ATTR,
          sizeof(MIPI_ATTR_S));
      break;
    case kAR0233_30FPS_1080P_RAW12_954_PWL:
      memcpy(pst_mipi_attr,
          &MIPI_4LANE_SENSOR_AR0233_30FPS_12BIT_1080P_954_ATTR,
          sizeof(MIPI_ATTR_S));
      break;
    case kAR0233_30FPS_1080P_RAW12_960_PWL:
      memcpy(pst_mipi_attr,
          &MIPI_4LANE_SENSOR_AR0233_30FPS_12BIT_1080P_960_ATTR,
          sizeof(MIPI_ATTR_S));
      break;
    case kOS8A10_30FPS_3840P_RAW10_LINEAR:
      if (need_clk == 1) {
        memcpy(pst_mipi_attr,
            &MIPI_SENSOR_OS8A10_30FPS_10BIT_LINEAR_SENSOR_CLK_ATTR,
            sizeof(MIPI_ATTR_S));
      } else {
        memcpy(pst_mipi_attr,
            &MIPI_SENSOR_OS8A10_30FPS_10BIT_LINEAR_ATTR, sizeof(MIPI_ATTR_S));
      }
      break;
    case kOS8A10_30FPS_3840P_RAW10_DOL2:
      memcpy(pst_mipi_attr,
          &MIPI_SENSOR_OS8A10_30FPS_10BIT_DOL2_ATTR,
          sizeof(MIPI_ATTR_S));
      break;
    case kOV10635_30FPS_720p_954_YUV:
      memcpy(pst_mipi_attr,
          &MIPI_2LANE_OV10635_30FPS_YUV_720P_954_ATTR,
          sizeof(MIPI_ATTR_S));
      break;
    case kOV10635_30FPS_720p_960_YUV:
      memcpy(pst_mipi_attr,
          &MIPI_2LANE_OV10635_30FPS_YUV_720P_960_ATTR,
          sizeof(MIPI_ATTR_S));
      break;
    case kOV10635_30FPS_720p_960_YUV_LINE_CONCATENATED:
      memcpy(pst_mipi_attr,
          &MIPI_2LANE_OV10635_30FPS_YUV_LINE_CONCATE_720P_960_ATTR,
          sizeof(MIPI_ATTR_S));
      break;
    case kS5KGM1SP_30FPS_4000x3000_RAW10:
      memcpy(pst_mipi_attr,
          &MIPI_SENSOR_S5KGM1SP_30FPS_10BIT_LINEAR_ATTR,
          sizeof(MIPI_ATTR_S));
      break;
    case kF37_30FPS_1080P_RAW10_LINEAR:
      if (need_clk == 1) {
        memcpy(pst_mipi_attr,
            &MIPI_SENSOR_F37_30FPS_10BIT_LINEAR_SENSOR_CLK_ATTR,
            sizeof(MIPI_ATTR_S));
      } else {
        memcpy(pst_mipi_attr,
            &MIPI_SENSOR_F37_30FPS_10BIT_LINEAR_ATTR, sizeof(MIPI_ATTR_S));
      }
      break;
    case kSC8238_30FPS_3840P_RAW10_LINEAR:
      if (need_clk == 1) {
        memcpy(pst_mipi_attr,
               &MIPI_SENSOR_SC8238_30FPS_10BIT_LINEAR_SENSOR_CLK_ATTR,
               sizeof(MIPI_ATTR_S));
      } else {
        memcpy(pst_mipi_attr,
               &MIPI_SENSOR_SC8238_30FPS_10BIT_LINEAR_ATTR,
               sizeof(MIPI_ATTR_S));
      }
      break;
    case kSC8238_30FPS_3840P_RAW10_DOL2:
      memcpy(pst_mipi_attr,
             &MIPI_SENSOR_SC8238_30FPS_10BIT_DOL2_ATTR,
             sizeof(MIPI_ATTR_S));
      break;
    case kGC4C33_30FPS_1440P_LINEAR:
      if (need_clk == 1) {
        memcpy(pst_mipi_attr,
               &MIPI_SENSOR_GC4C33_30FPS_1440P_LINEAR_SENSOR_CLK_ATTR,
               sizeof(MIPI_ATTR_S));
      } else {
        memcpy(pst_mipi_attr,
         &MIPI_SENSOR_GC4C33_30FPS_1440P_LINEAR_ATTR, sizeof(MIPI_ATTR_S));
      }
      break;
    default:
      LOGE << "not support sensor type";
      break;
  }
  LOGD << "get mipi host attr success...";

  return 0;
}

int VinModule::HbVinGetDevAttrBySns(MipiSensorType sensor_type,
    VIN_DEV_ATTR_S *pstDevAttr) {

  switch (sensor_type) {
    case kIMX327_30FPS_1952P_RAW12_LINEAR:
      memcpy(pstDevAttr, &DEV_ATTR_IMX327_LINEAR_BASE,
          sizeof(VIN_DEV_ATTR_S));
      break;
    case kIMX327_30FPS_2228P_RAW12_DOL2:
      memcpy(pstDevAttr, &DEV_ATTR_IMX327_DOL2_BASE,
          sizeof(VIN_DEV_ATTR_S));
      break;
    case kAR0233_30FPS_1080P_RAW12_954_PWL:
    case kAR0233_30FPS_1080P_RAW12_960_PWL:
      memcpy(pstDevAttr, &DEV_ATTR_AR0233_1080P_BASE,
          sizeof(VIN_DEV_ATTR_S));
      break;
    case kOS8A10_30FPS_3840P_RAW10_LINEAR:
      memcpy(pstDevAttr, &DEV_ATTR_OS8A10_LINEAR_BASE,
          sizeof(VIN_DEV_ATTR_S));
      break;
    case kOS8A10_30FPS_3840P_RAW10_DOL2:
      memcpy(pstDevAttr, &DEV_ATTR_OS8A10_DOL2_BASE,
          sizeof(VIN_DEV_ATTR_S));
      break;
    case kOV10635_30FPS_720p_954_YUV:
    case kOV10635_30FPS_720p_960_YUV:
      memcpy(pstDevAttr, &DEV_ATTR_OV10635_YUV_BASE,
          sizeof(VIN_DEV_ATTR_S));
      break;
    case kOV10635_30FPS_720p_960_YUV_LINE_CONCATENATED:
      memcpy(pstDevAttr, &DEV_ATTR_OV10635_YUV_LINE_CONCATE_BASE,
          sizeof(VIN_DEV_ATTR_S));
      break;
    case kSIF_TEST_PATTERN_1080P:
      memcpy(pstDevAttr, &DEV_ATTR_TEST_PATTERN_1080P_BASE,
          sizeof(VIN_DEV_ATTR_S));
      break;
    case kSIF_TEST_PATTERN_4K:
      memcpy(pstDevAttr, &DEV_ATTR_TEST_PATTERN_4K_BASE,
          sizeof(VIN_DEV_ATTR_S));
      break;
    case kFEED_BACK_RAW12_1952P:
      memcpy(pstDevAttr, &DEV_ATTR_FEED_BACK_1097P_BASE,
          sizeof(VIN_DEV_ATTR_S));
      break;
    case kSIF_TEST_PATTERN_YUV_720P:
      memcpy(pstDevAttr, &DEV_ATTR_TEST_PATTERN_YUV422_BASE,
          sizeof(VIN_DEV_ATTR_S));
      break;
    case kSIF_TEST_PATTERN_12M_RAW12:
      memcpy(pstDevAttr, &DEV_ATTR_TEST_PATTERN_12M_BASE,
          sizeof(VIN_DEV_ATTR_S));
      break;
    case kS5KGM1SP_30FPS_4000x3000_RAW10:
      memcpy(pstDevAttr, &DEV_ATTR_S5KGM1SP_LINEAR_BASE,
          sizeof(VIN_DEV_ATTR_S));
      break;
    case kF37_30FPS_1080P_RAW10_LINEAR:
      memcpy(pstDevAttr, &DEV_ATTR_F37_LINEAR_BASE,
          sizeof(VIN_DEV_ATTR_S));
      break;
    case kSC8238_30FPS_3840P_RAW10_LINEAR:
      memcpy(pstDevAttr, &DEV_ATTR_SC8238_LINEAR_BASE,
             sizeof(VIN_DEV_ATTR_S));
      break;
    case kGC4C33_30FPS_1440P_LINEAR:
      memcpy(pstDevAttr, &DEV_ATTR_GC4C33_LINEAR_BASE,
             sizeof(VIN_DEV_ATTR_S));
      break;
    default:
      LOGE << "not surpport sensor type";
      break;
  }
  LOGD << "get vin dev attr success...";

  return 0;
}

int VinModule::HbVinGetDevAttrExBySns(MipiSensorType sensor_type,
    VIN_DEV_ATTR_EX_S *pstDevAttrEx) {

  switch (sensor_type) {
    case kIMX327_30FPS_1952P_RAW12_LINEAR:
      memcpy(pstDevAttrEx, &DEV_ATTR_IMX327_MD_BASE,
          sizeof(VIN_DEV_ATTR_EX_S));
      break;
    case kOV10635_30FPS_720p_960_YUV:
    case kOV10635_30FPS_720p_954_YUV:
      memcpy(pstDevAttrEx, &DEV_ATTR_OV10635_MD_BASE,
          sizeof(VIN_DEV_ATTR_EX_S));
      break;

    default:
      LOGW << "not surpport sensor type";
      break;
  }
  LOGD << "Get vin dev external attr success...";

  return 0;
}

int VinModule::HbVinGetPipeAttrBySns(MipiSensorType sensor_type,
    VIN_PIPE_ATTR_S *pstPipeAttr) {

  switch (sensor_type) {
    case kIMX327_30FPS_1952P_RAW12_LINEAR:
    case kFEED_BACK_RAW12_1952P:
      memcpy(pstPipeAttr, &PIPE_ATTR_IMX327_LINEAR_BASE,
          sizeof(VIN_PIPE_ATTR_S));
      break;
    case kIMX327_30FPS_2228P_RAW12_DOL2:
      memcpy(pstPipeAttr, &PIPE_ATTR_IMX327_DOL2_BASE,
          sizeof(VIN_PIPE_ATTR_S));
      break;
    case kAR0233_30FPS_1080P_RAW12_954_PWL:
    case kAR0233_30FPS_1080P_RAW12_960_PWL:
      memcpy(pstPipeAttr, &PIPE_ATTR_AR0233_1080P_BASE,
          sizeof(VIN_PIPE_ATTR_S));
      break;
    case kOS8A10_30FPS_3840P_RAW10_LINEAR:
      memcpy(pstPipeAttr, &PIPE_ATTR_OS8A10_LINEAR_BASE,
          sizeof(VIN_PIPE_ATTR_S));
      break;
    case kOS8A10_30FPS_3840P_RAW10_DOL2:
      memcpy(pstPipeAttr, &PIPE_ATTR_OS8A10_DOL2_BASE,
          sizeof(VIN_PIPE_ATTR_S));
      break;
    case kOV10635_30FPS_720p_954_YUV:
    case kOV10635_30FPS_720p_960_YUV:
    case kSIF_TEST_PATTERN_YUV_720P:
      memcpy(pstPipeAttr, &PIPE_ATTR_OV10635_YUV_BASE,
          sizeof(VIN_PIPE_ATTR_S));
      break;
    case kOV10635_30FPS_720p_960_YUV_LINE_CONCATENATED:
      memcpy(pstPipeAttr, &VIN_ATTR_OV10635_YUV_LINE_CONCATE_BASE ,
          sizeof(VIN_PIPE_ATTR_S));
      break;
    case kSIF_TEST_PATTERN_1080P:
      memcpy(pstPipeAttr, &PIPE_ATTR_TEST_PATTERN_1080P_BASE,
          sizeof(VIN_PIPE_ATTR_S));
      break;
    case kSIF_TEST_PATTERN_4K:
      memcpy(pstPipeAttr, &PIPE_ATTR_TEST_PATTERN_4K_BASE,
          sizeof(VIN_PIPE_ATTR_S));
      break;
    case kSIF_TEST_PATTERN_12M_RAW12:
      memcpy(pstPipeAttr, &PIPE_ATTR_TEST_PATTERN_12M_BASE,
          sizeof(VIN_PIPE_ATTR_S));
      break;
    case kS5KGM1SP_30FPS_4000x3000_RAW10:
      memcpy(pstPipeAttr, &PIPE_ATTR_S5KGM1SP_LINEAR_BASE,
          sizeof(VIN_PIPE_ATTR_S));
      break;
    case kF37_30FPS_1080P_RAW10_LINEAR:
      memcpy(pstPipeAttr, &PIPE_ATTR_F37_LINEAR_BASE,
          sizeof(VIN_PIPE_ATTR_S));
      break;
    case kSC8238_30FPS_3840P_RAW10_LINEAR:
      memcpy(pstPipeAttr, &PIPE_ATTR_SC8238_LINEAR_BASE,
             sizeof(VIN_PIPE_ATTR_S));
      break;
    case kGC4C33_30FPS_1440P_LINEAR:
      memcpy(pstPipeAttr, &PIPE_ATTR_GC4C33_LINEAR_BASE,
             sizeof(VIN_PIPE_ATTR_S));
      break;
    default:
      LOGE << "not surpport sensor type";
      break;
  }
  LOGD << "Get vin pipe attr success...";

  return 0;
}

int VinModule::HbVinGetDisAttrBySns(MipiSensorType sensor_type,
    VIN_DIS_ATTR_S *pstDisAttr) {

  switch (sensor_type) {
    case kIMX327_30FPS_1952P_RAW12_LINEAR:
    case kIMX327_30FPS_2228P_RAW12_DOL2:
    case kAR0233_30FPS_1080P_RAW12_954_PWL:
    case kAR0233_30FPS_1080P_RAW12_960_PWL:
    case kSIF_TEST_PATTERN_1080P:
    case kSIF_TEST_PATTERN_4K:
    case kFEED_BACK_RAW12_1952P:
      memcpy(pstDisAttr, &DIS_ATTR_BASE, sizeof(VIN_DIS_ATTR_S));
      break;
    case kOS8A10_30FPS_3840P_RAW10_LINEAR:
    case kOS8A10_30FPS_3840P_RAW10_DOL2:
      memcpy(pstDisAttr, &DIS_ATTR_OS8A10_BASE, sizeof(VIN_DIS_ATTR_S));
      break;
    case kOV10635_30FPS_720p_954_YUV:
    case kOV10635_30FPS_720p_960_YUV:
    case kOV10635_30FPS_720p_960_YUV_LINE_CONCATENATED:
    case kSIF_TEST_PATTERN_YUV_720P:
      memcpy(pstDisAttr, &DIS_ATTR_OV10635_BASE, sizeof(VIN_DIS_ATTR_S));
      break;
    case kSIF_TEST_PATTERN_12M_RAW12:
    case kS5KGM1SP_30FPS_4000x3000_RAW10:
      memcpy(pstDisAttr, &DIS_ATTR_12M_BASE, sizeof(VIN_DIS_ATTR_S));
      break;
    case kSC8238_30FPS_3840P_RAW10_LINEAR:
      memcpy(pstDisAttr, &DIS_ATTR_SC8238_BASE, sizeof(VIN_DIS_ATTR_S));
      break;
    default:
      LOGE << "not surpport sensor type";
      break;
  }
  LOGD << "Get vin dis attr success...";

  return 0;
}

int VinModule::HbVinGetLdcAttrBySns(MipiSensorType sensor_type,
    VIN_LDC_ATTR_S *pstLdcAttr) {

  switch (sensor_type) {
    case kIMX327_30FPS_1952P_RAW12_LINEAR:
    case kIMX327_30FPS_2228P_RAW12_DOL2:
    case kAR0233_30FPS_1080P_RAW12_954_PWL:
    case kAR0233_30FPS_1080P_RAW12_960_PWL:
    case kSIF_TEST_PATTERN_1080P:
    case kSIF_TEST_PATTERN_4K:
    case kFEED_BACK_RAW12_1952P:
      memcpy(pstLdcAttr, &LDC_ATTR_BASE, sizeof(VIN_LDC_ATTR_S));
      break;
    case kOS8A10_30FPS_3840P_RAW10_LINEAR:
    case kOS8A10_30FPS_3840P_RAW10_DOL2:
      memcpy(pstLdcAttr, &LDC_ATTR_OS8A10_BASE, sizeof(VIN_LDC_ATTR_S));
      break;
    case kOV10635_30FPS_720p_954_YUV:
    case kOV10635_30FPS_720p_960_YUV:
    case kOV10635_30FPS_720p_960_YUV_LINE_CONCATENATED:
    case kSIF_TEST_PATTERN_YUV_720P:
      memcpy(pstLdcAttr, &LDC_ATTR_OV10635_BASE, sizeof(VIN_LDC_ATTR_S));
      break;
    case kSIF_TEST_PATTERN_12M_RAW12:
    case kS5KGM1SP_30FPS_4000x3000_RAW10:
      memcpy(pstLdcAttr, &LDC_ATTR_12M_BASE, sizeof(VIN_LDC_ATTR_S));
      break;
    case kSC8238_30FPS_3840P_RAW10_LINEAR:
      memcpy(pstLdcAttr, &LDC_ATTR_SC8238_BASE, sizeof(VIN_LDC_ATTR_S));
      break;
    default:
      LOGE << "not surpport sensor type";
      break;
  }
  LOGD << "Get vin ldc attr success...";

  return 0;
}

int VinModule::HbTimeCostMs(struct timeval *start, struct timeval *end) {
  int time_ms = -1;
  time_ms = ((end->tv_sec * 1000 + end->tv_usec / 1000) -
      (start->tv_sec * 1000 + start->tv_usec / 1000));
  LOGD << "time cost " << time_ms << "ms";
  return time_ms;
}

void VinModule::HbPrintSensorDevInfo(VIN_DEV_ATTR_S *devinfo) {
  LOGI << "sensor format:" << devinfo->stSize.format;
  LOGI << "sensor width:" << devinfo->stSize.width;
  LOGI << "sensor height:" << devinfo->stSize.height;
  LOGI << "sensor pix_length:" << devinfo->stSize.pix_length;
  LOGI << "sensor mipi attr enable_frame_id:"
    << devinfo->mipiAttr.enable_frame_id;
  LOGI << "sensor mipi attr enable_mux_out:"
    << devinfo->mipiAttr.enable_mux_out;
  LOGI << "sensor mipi attr set_init_frame_id:"
    << devinfo->mipiAttr.set_init_frame_id;
  LOGI << "sensor mipi attr ipu_channels:"
    << devinfo->mipiAttr.ipi_channels;
  LOGI << "sensor mipi attr enable_line_shift:"
    << devinfo->mipiAttr.enable_line_shift;
  LOGI << "sensor mipi attr enable_id_decoder :"
    << devinfo->mipiAttr.enable_id_decoder;
  LOGI << "sensor mipi attr set_bypass_channels :"
    << devinfo->mipiAttr.set_bypass_channels;
  LOGI << "sensor mipi attr enable_bypass :"
    << devinfo->mipiAttr.enable_bypass;
  LOGI << "sensor mipi attr set_line_shift_count :"
    << devinfo->mipiAttr.set_line_shift_count;
  LOGI << "sensor mipi attr enable_pattern :"
    << devinfo->mipiAttr.enable_pattern;

  LOGI << "sensor out ddr attr stride :"
    << devinfo->outDdrAttr.stride;
  LOGI << "sensor out ddr attr buffer_num :"
    << devinfo->outDdrAttr.buffer_num;

  return;
}

void VinModule::HbPrintSensorPipeInfo(VIN_PIPE_ATTR_S *pipeinfo) {
  LOGI << "isp_out ddr_out_buf_num: " << pipeinfo->ddrOutBufNum;
  LOGI << "isp_out width: " << pipeinfo->stSize.width;
  LOGI << "isp_out height: " << pipeinfo->stSize.height;
  LOGI << "isp_out sensor_mode: " << pipeinfo->snsMode;
  LOGI << "isp_out format: " << pipeinfo->stSize.format;

  return;
}

void VinModule::HbPrintSensorInfo(MIPI_SENSOR_INFO_S *snsinfo) {
  LOGI << "bus_bum: " << snsinfo->sensorInfo.bus_num;
  LOGI << "bus_type: " << snsinfo->sensorInfo.bus_type;
  LOGI << "reg_width: " << snsinfo->sensorInfo.reg_width;
  LOGI << "sensor_name: " << snsinfo->sensorInfo.sensor_name;
  LOGI << "sensor_mode: " << snsinfo->sensorInfo.sensor_mode;
  LOGI << "sensor_addr: " << snsinfo->sensorInfo.sensor_addr;
  LOGI << "serial_addr: " << snsinfo->sensorInfo.serial_addr;
  LOGI << "resolution: " << snsinfo->sensorInfo.resolution;
  LOGI << "gpio_num: " << snsinfo->sensorInfo.gpio_num;
  LOGI << "gpio_pin: " << snsinfo->sensorInfo.gpio_pin[0];
  LOGI << "gpio_level: " << snsinfo->sensorInfo.gpio_level[0];

  return;
}

void VinModule::HbVinSetConfig() {
  int temper_mode, isp_out_buf_num, ldc_en, dis_en;

  /* 1. get vin config from vio_cfg */
  temper_mode = vio_cfg_.vin_cfg.isp_info.temper_mode;
  isp_out_buf_num = vio_cfg_.vin_cfg.isp_info.isp_out_buf_num;
  ldc_en      = vio_cfg_.vin_cfg.dwe_info.ldc_en;
  dis_en      = vio_cfg_.vin_cfg.dwe_info.dis_en;
  /* 2. set vin config to vin_params */
  vin_params_.pipe_info.temperMode = temper_mode;
  vin_params_.pipe_info.ddrOutBufNum = isp_out_buf_num;
  vin_params_.ldc_info.ldcEnable = ldc_en;
  vin_params_.dis_info.disPath.rg_dis_enable = dis_en;

  return;
}

}  // namespace vioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
