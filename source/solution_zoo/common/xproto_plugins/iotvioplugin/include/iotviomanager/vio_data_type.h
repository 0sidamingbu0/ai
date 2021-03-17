/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2020 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#ifndef INCLUDE_VIO_DATA_TYPE_H_
#define INCLUDE_VIO_DATA_TYPE_H_
#include <string>
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
#include <stdint.h>
#include <string.h>
#include "hb_sys.h"
#include "hb_mipi_api.h"
#include "hb_vin_api.h"
#include "hb_vio_interface.h"
#include "hb_vps_api.h"
#include "hb_vp_api.h"
#ifdef __cplusplus
}
#endif  /* __cplusplus */

/* max camera num */
#define MAX_GDC_NUM                              (2)
#define MAX_MIPIID_NUM                           (4)
#define MAX_CHN_NUM                              (7)
#define MAX_PIPE_NUM                             (6)
#define MAX_POOL_CNT                             (32)
#define GDC_PIPE_ID                              (MAX_PIPE_NUM+1)
#define TIME_MICRO_SECOND                        (1000000)
#define __FILENAME__ (strrchr(__FILE__, '/') ?\
        (strrchr(__FILE__, '/') + 1):__FILE__)

#define TYPE_NAME(x)    #x
/* for get info type */
#define IOT_VIO_SRC_INFO                         (1)
#define IOT_VIO_PYM_INFO                         (2)
#define IOT_VIO_SRC_MULT_INFO                    (3)
#define IOT_VIO_PYM_MULT_INFO                    (4)
#define IOT_VIO_FEEDBACK_SRC_INFO                (5)
#define IOT_VIO_FEEDBACK_FLUSH                   (6)
#define IOT_VIO_FEEDBACK_PYM_PROCESS             (7)
#define IOT_VIO_CHN_INFO                         (8)

#define CHECK_PARAMS_VALID(p)\
  do {\
    if (p == NULL)\
    return -1;\
  } while (0)

typedef enum MipiSensorType {
  kSENSOR_ID_INVALID = 0,
  kIMX327_30FPS_1952P_RAW12_LINEAR,   // 1
  kIMX327_30FPS_2228P_RAW12_DOL2,     // 2
  kAR0233_30FPS_1080P_RAW12_954_PWL,  // 3
  kAR0233_30FPS_1080P_RAW12_960_PWL,  // 4
  kOS8A10_30FPS_3840P_RAW10_LINEAR,   // 5
  kOS8A10_30FPS_3840P_RAW10_DOL2,     // 6
  kOV10635_30FPS_720p_954_YUV,        // 7
  kOV10635_30FPS_720p_960_YUV,        // 8
  kSIF_TEST_PATTERN_1080P,           // 9
  kFEED_BACK_RAW12_1952P,             // 10
  kSIF_TEST_PATTERN_YUV_720P,         // 11
  kSIF_TEST_PATTERN_12M_RAW12,        // 12
  kSIF_TEST_PATTERN_4K,               // 13
  kS5KGM1SP_30FPS_4000x3000_RAW10,    // 14
  kOV10635_30FPS_720p_960_YUV_LINE_CONCATENATED,  // 15
  kF37_30FPS_1080P_RAW10_LINEAR,      // 16
  kSC8238_30FPS_3840P_RAW10_LINEAR,   // 17
  kSC8238_30FPS_3840P_RAW10_DOL2,     // 18
  kGC4C33_30FPS_1440P_LINEAR,     // 19
  kSAMPLE_SENOSR_ID_MAX
} MipiSensorTypeE;

typedef enum GdcType {
  kGDC_TYPE_INVALID = 0,
  kISP_DDR_GDC,
  kIPU_CHN_DDR_GDC,
  kPYM_DDR_GDC,
  kGDC_TYPE_MAX
} IotGdcType_E;

typedef enum VioSourceType {
  kVIO_SOURCE_INVALID = 0,
  kCAM_VIO_SOURCE,
  kFB_VIO_SOURCE,
  kVIO_SOURCE_MAX
} IotVioSrcType_E;

typedef enum FbSourceType {
  kFB_SOURCE_INVALID = 0,
  kSIF_BUF_SOURCE,
  kISP_BUF_SOURCE,
  kFB_SOURCE_MAX
} IotFbSrcType_E;

typedef struct VioChnCfg {
  int ipu_chn_num;
  int pym_chn_num;
  int ipu_valid_chn[MAX_CHN_NUM];
  int pym_valid_chn[MAX_CHN_NUM];
} IotVioChnInfo;

typedef struct VioRoiInfo {
  int roi_en;
  int roi_x;
  int roi_y;
  int roi_w;
  int roi_h;
} IotVioRoiInfo;

typedef struct VioScaleInfo {
  int scale_en;
  int scale_w;
  int scale_h;
} IotVioScaleInfo;

typedef struct VioGdcInfo {
  int gdc_en;
  int gdc_w;
  int gdc_h;
  int frame_depth;
  int src_type;
  int rotate;
  int bind_ipu_chn;
  std::string file_path;
} IotVioGdcInfo;

typedef struct VioGrpInfo {
  int width;
  int height;
  int frame_depth;
} IotVioGrpInfo;

typedef struct VioSensorInfo {
  int sensor_id;
  int plugin_en;
  std::string sensor_path;
  std::string sensor_name;
  int sensor_type;
  int sensor_port;
  int i2c_bus;
  int need_clk;
  int serdes_index;
  int serdes_port;
  int extra_mode;
  int gpio_num;
  int gpio_pin;
  int gpio_level;
} IotVioSensorInfo;

typedef struct VioMipiInfo {
  int mipi_index;
  int vc_index;
  int dol2_vc_index;
} IotVioMipiInfo;

typedef struct VioSifInfo {
  int need_md;
  int sif_out_buf_num;
} IotVioSifInfo;

typedef struct VioIspInfo {
  int temper_mode;
  int isp_out_buf_num;
  int isp_3a_en;
} IotVioIspInfo;

typedef struct VioDweInfo {
  int ldc_en;
  int dis_en;
} IotVioDweInfo;

typedef struct VinInfo {
  int vin_fd_en;
  // sensor config
  IotVioSensorInfo sensor_info;
  // mipi config
  IotVioMipiInfo mipi_info;
  // sif config
  IotVioSifInfo sif_info;
  // isp config
  IotVioIspInfo isp_info;
  // dwe config
  IotVioDweInfo dwe_info;
} IotVinInfo;

typedef struct ChnAttr {
  int ipu_chn_en;
  int pym_chn_en;
  int ipu_frame_depth;
  int ipu_frame_timeout;
  IotVioRoiInfo ipu_roi;
  IotVioScaleInfo ipu_scale;
} IotVioChnAttr;

typedef struct FeedbackInfo {
  int width;
  int height;
  int buf_num;
  int inner_buf_en;
  int inner_buf_type;
  int bind_pipe_id;
} IotVioFbInfo;

typedef struct VpsInfo {
  int vps_dump_num;
  int vps_layer_dump;
  int vps_fd_en;
  // feedback config
  IotVioFbInfo fb_info;
  // gdc config
  IotVioGdcInfo gdc_info[MAX_GDC_NUM];
} IotVpsInfo;

typedef struct VioInfo {
  int pipe_id;
  int chn;
  int buf_index;
  hb_vio_buffer_t vio_buf;
} IotVioInfo;

typedef struct PymInfo {
  int grp_id;
  int pym_chn;
  int buf_index;
  pym_buffer_t pym_buf;
} IotPymInfo;

#if 0
typedef struct MultSrcInfo {
  int pipe_num;
  IotVioChnInfo chn_info[MAX_PIPE_NUM];
  hb_vio_buffer_t vinfo[MAX_PIPE_NUM][MAX_CHN_NUM];
} IotMultSrcBuffer;

typedef struct MultPymInfo {
  int pipe_num;
  pym_buffer_t pinfo[MAX_PIPE_NUM];
} IotMultPymBuffer;
#else
typedef struct MultSrcInfo {
  int img_num;
  hb_vio_buffer_t img_info[MAX_PIPE_NUM];
} IotMultSrcBuffer;

typedef struct MultPymInfo {
  int img_num;
  pym_buffer_t img_info[MAX_PIPE_NUM];
} IotMultPymBuffer;
#endif

typedef struct VioCfg {
  // 1.vin vps config
  int cam_en;
  int vin_vps_mode;
  // 2.vin detail config
  IotVinInfo vin_cfg;
  // 3.vps detail config
  IotVpsInfo vps_cfg;
  // 4.ipu chn config
  int vps_chn_num;
  IotVioChnAttr chn_cfg[MAX_CHN_NUM];
  // 5.pym chn config
  int pym_chn_index;
  VPS_PYM_CHN_ATTR_S pym_cfg;
} IotVioCfg;

typedef struct CamParams {
  MIPI_SENSOR_INFO_S sensor_info;
  MIPI_ATTR_S mipi_info;
} IotCamParams;

typedef struct VinParams {
  VIN_DEV_ATTR_S dev_info;
  VIN_DEV_ATTR_EX_S devex_info;
  VIN_PIPE_ATTR_S pipe_info;
  VIN_DIS_ATTR_S dis_info;
  VIN_LDC_ATTR_S ldc_info;
} IotVinParams;

#endif  // INCLUDE_VIO_DATA_TYPE_H
