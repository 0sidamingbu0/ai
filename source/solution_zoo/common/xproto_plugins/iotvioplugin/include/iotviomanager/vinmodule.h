/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2020 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#ifndef INCLUDE_VIN_MODULE_H_
#define INCLUDE_VIN_MODULE_H_

#include <sys/time.h>
#include <sys/types.h>
#include "iotviomanager/vio_data_type.h"
#include "iotviomanager/sensor_plugin.h"
namespace horizon {
namespace vision {
namespace xproto {
namespace vioplugin {

class VinModule {
 public:
  VinModule() = delete;
  VinModule(const int &pipe_id, const IotVioCfg &vio_cfg)
    : pipe_id_(pipe_id), vio_cfg_(vio_cfg) {}
  ~VinModule() {}

  int Init();
  int DeInit();
  int Start();
  int Stop();
  int CamInit();
  int CamDeInit();
  int CamStart();
  int CamStop();
  int VinInit();
  int VinDeInit();
  int VinStart();
  int VinStop();
  int GetVinConfig(IotVinParams *vin_cfg);
  int GetCamConfig(IotVinParams *cam_cfg);

 private:
  int HbMipiGetSnsAttrBySns(MipiSensorType sensor_type,
      MIPI_SENSOR_INFO_S *pst_sns_attr);
  int HbMipiGetMipiAttrBySns(MipiSensorType sensor_type,
      MIPI_ATTR_S *pst_mipi_attr);
  int HbVinGetDevAttrBySns(MipiSensorType sensor_type,
      VIN_DEV_ATTR_S *pstDevAttr);
  int HbVinGetDevAttrExBySns(MipiSensorType sensor_type,
      VIN_DEV_ATTR_EX_S *pstDevAttrEx);
  int HbVinGetPipeAttrBySns(MipiSensorType sensor_type,
      VIN_PIPE_ATTR_S *pstPipeAttr);
  int HbVinGetDisAttrBySns(MipiSensorType sensor_type,
      VIN_DIS_ATTR_S *pstDisAttr);
  int HbVinGetLdcAttrBySns(MipiSensorType sensor_type,
      VIN_LDC_ATTR_S *pstLdcAttr);
  int HbTimeCostMs(struct timeval *start, struct timeval *end);
  void HbPrintSensorDevInfo(VIN_DEV_ATTR_S *devinfo);
  void HbPrintSensorPipeInfo(VIN_PIPE_ATTR_S *pipeinfo);
  void HbPrintSensorInfo(MIPI_SENSOR_INFO_S *snsinfo);
  void HbVinSetConfig();
  int HbSensorInit(int dev_id, int sensor_id, int bus, int port,
      int mipi_idx, int sedres_index, int sedres_port);
  int HbSensorDeInit(int pipe_id);
  int HbSensorStart(int pipe_id);
  int HbSensorStop(int pipe_id);
  int HbEnableSensorClk(int mipi_idx);
  int HbDisableSensorClk(int mipi_idx);
  int HbMipiInit(int sensor_id, int mipi_idx);
  int HbMipiDeInit(int mipi_idx);
  int HbMipiStart(int mipi_idx);
  int HbMipiStop(int mipi_idx);
  int HbSensorPlgInit();
  int HbSensorPlgDeInit();
  static void HbDisCropSet(uint32_t pipe_id, uint32_t event,
      VIN_DIS_MV_INFO_S *data, void *userdata);


 private:
  int pipe_id_;
  IotVioCfg    vio_cfg_;
  IotVinParams vin_params_ = { 0 };
  IotCamParams cam_params_ = { 0 };
  int vin_fd_ = -1;
  bool vin_init_flag_ = false;
  bool cam_init_flag_ = false;
  bool sensor_plg_en_ = false;
  SnsPlgCtrlInfo sensor_plg_cfg_ = { 0 };
};

}  // namespace vioplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
#endif  // INCLUDE_VIN_MODULE_H_
