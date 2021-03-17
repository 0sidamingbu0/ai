/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2020 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#include <iostream>
#include "iotviomanager/sensor_plugin.h"

MIPI_SENSOR_INFO_S SENSOR_GC4C33_30FPS_1080P_LINEAR_INFO = {
    .deseEnable = 0,
    .inputMode = INPUT_MODE_MIPI,
    .deserialInfo = {},
    .sensorInfo = {
        .port = 0,
        .dev_port = 0,
        .bus_type = 0,
        .bus_num = 2,
        .fps = 30,
        .resolution = 1440,
        .sensor_addr = 0x29,
        .serial_addr = 0,
        .entry_index = 0,
        .sensor_mode = static_cast<MIPI_SENSOR_MODE_E>(0),
        .reg_width = 16,
        .sensor_name = const_cast<char*>("gc4c33"),
        .extra_mode = 0,
        .deserial_index = 0,
        .deserial_port = 0,
    }
};

MIPI_ATTR_S MIPI_SENSOR_GC4C33_30FPS_1080P_LINEAR_ATTR = {
    .mipi_host_cfg = {
        2,           /* lane */
        0x2b,        /* datatype */
        24,          /* mclk */
        630,        /* mipiclk */
        30,          /* fps */
        2560,        /* width */
        1440,        /*height */
        2720,        /* linlength */
        1500,        /* framelength */
        30,          /* settle */
        4,           /*chnnal_num*/
        {0, 1, 2, 3} /*vc */
    },
    .dev_enable = 0 /*  mipi dev enable */
};

MIPI_ATTR_S MIPI_SENSOR_GC4C33_30FPS_1080P_LINEAR_SENSOR_CLK_ATTR = {
    .mipi_host_cfg = {
        2,           /* lane */
        0x2b,        /* datatype */
        2400,        /* mclk */
        630,        /* mipiclk */
        30,          /* fps */
        2560,        /* width  */
        1440,        /*height */
        2720,        /* linlength */
        1500,        /* framelength */
        30,          /* settle */
        4,           /*chnnal_num*/
        {0, 1, 2, 3} /*vc */
    },
    .dev_enable = 0 /*  mipi dev enable */
};

VIN_DEV_ATTR_S DEV_ATTR_GC4C33_LINEAR_BASE = {
    .stSize = {
        0,    /*format*/
        2560, /*width*/
        1440, /*height*/
        1     /*pix_length*/
    },
    {
    .mipiAttr =
        {
            .enable = 1,
            .ipi_channels = 1,
            .ipi_mode = 0,
            .enable_mux_out = 1,
            .enable_frame_id = 1,
            .enable_bypass = 0,
            .enable_line_shift = 0,
            .enable_id_decoder = 0,
            .set_init_frame_id = 1,
            .set_line_shift_count = 0,
            .set_bypass_channels = 1,
        },
    },
    .DdrIspAttr = {.stride = 0,
                   .buf_num = 4,
                   .raw_feedback_en = 0,
                   .data =
                       {
                           .format = 0,
                           .width = 2560,
                           .height = 1440,
                           .pix_length = 1,
                       }},
    .outDdrAttr =
        {
            .stride = 2400,
            .buffer_num = 8,
        },
    .outIspAttr = {
        .dol_exp_num = 1,
        .enable_dgain = 0,
        .set_dgain_short = 0,
        .set_dgain_medium = 0,
        .set_dgain_long = 0,
    }};

VIN_PIPE_ATTR_S PIPE_ATTR_GC4C33_LINEAR_BASE = {
    .ddrOutBufNum = 8,
    .frameDepth = 0,
    .snsMode = SENSOR_NORMAL_MODE,
    .stSize = {
        .format = 0,
        .width = 2560,
        .height = 1440,
    },
    .cfaPattern = static_cast<VIN_PIPE_CFA_PATTERN_E>(0),
    .temperMode = 2,
    .ispBypassEn = 0,
    .ispAlgoState = 1,
    .bitwidth = 10,
    .startX = 0,
    .startY = 0,
    .calib = {
        .mode = 1,
        .lname = const_cast<char*>("/etc/cam/libgc4c33_linear.so"),
    }
};

static int GetSensorAttr(SnsPlgCtrlInfo *ctrl_info,
    MIPI_SENSOR_INFO_S *pst_sns_attr) {
  int ret = -1;
  IotSensorType_E sensor_type =
    static_cast<IotSensorType_E>(ctrl_info->sensor_type);

  if (sensor_type == kLINEAR_SENSOR_TYPE) {
    std::cout << "gc4c33 sensor plugin type is linear" << std::endl;
    *pst_sns_attr = SENSOR_GC4C33_30FPS_1080P_LINEAR_INFO;
  } else if (sensor_type == kDOL2_SENSOR_TYPE) {
    std::cout << "sensor type is dol2" << std::endl;
  } else if (sensor_type == kDOL3_SENSOR_TYPE) {
    std::cout << "sensor type is dol3" << std::endl;
  } else {
    std::cout << "not support sensor type: " << sensor_type << std::endl;
    return ret;
  }
  return 0;
}

static int GetMipiAttr(SnsPlgCtrlInfo *ctrl_info,
    MIPI_ATTR_S *pst_mipi_attr) {
  int ret = -1;
  int need_clk = ctrl_info->need_clk;
  IotSensorType_E sensor_type =
    static_cast<IotSensorType_E>(ctrl_info->sensor_type);

  std::cout << "gc4c33 sensor plugin need_clk:" << need_clk << std::endl;
  if (sensor_type == kLINEAR_SENSOR_TYPE) {
    if (need_clk == 1) {
      *pst_mipi_attr = MIPI_SENSOR_GC4C33_30FPS_1080P_LINEAR_SENSOR_CLK_ATTR;
    } else {
      *pst_mipi_attr = MIPI_SENSOR_GC4C33_30FPS_1080P_LINEAR_ATTR;
    }
  } else if (sensor_type == kDOL2_SENSOR_TYPE) {
  } else if (sensor_type == kDOL3_SENSOR_TYPE) {
  } else {
    std::cout << "not support sensor type: " << sensor_type << std::endl;
    return ret;
  }
  return 0;
}

static int GetVinDevAttr(SnsPlgCtrlInfo *ctrl_info,
    VIN_DEV_ATTR_S *pst_dev_attr) {
  int ret = -1;
  IotSensorType_E sensor_type =
    static_cast<IotSensorType_E>(ctrl_info->sensor_type);

  if (sensor_type == kLINEAR_SENSOR_TYPE) {
    *pst_dev_attr = DEV_ATTR_GC4C33_LINEAR_BASE;
  } else if (sensor_type == kDOL2_SENSOR_TYPE) {
  } else if (sensor_type == kDOL3_SENSOR_TYPE) {
  } else {
    std::cout << "not support sensor type: " << sensor_type << std::endl;
    return ret;
  }
  return 0;
}

static int GetVinDevAttrEx(SnsPlgCtrlInfo *ctrl_info,
    VIN_DEV_ATTR_EX_S *pst_dev_attr_ex) {
  int ret = -1;
  IotSensorType_E sensor_type =
    static_cast<IotSensorType_E>(ctrl_info->sensor_type);

  if (sensor_type == kLINEAR_SENSOR_TYPE) {
  } else if (sensor_type == kDOL2_SENSOR_TYPE) {
  } else if (sensor_type == kDOL3_SENSOR_TYPE) {
  } else {
    std::cout << "not support sensor type: " << sensor_type << std::endl;
    return ret;
  }
  return 0;
}

static int GetVinPipeAttr(SnsPlgCtrlInfo *ctrl_info,
     VIN_PIPE_ATTR_S *pst_pipe_attr) {
  int ret = -1;
  IotSensorType_E sensor_type =
    static_cast<IotSensorType_E>(ctrl_info->sensor_type);

  if (sensor_type == kLINEAR_SENSOR_TYPE) {
    *pst_pipe_attr = PIPE_ATTR_GC4C33_LINEAR_BASE;
  } else if (sensor_type == kDOL2_SENSOR_TYPE) {
  } else if (sensor_type == kDOL3_SENSOR_TYPE) {
  } else {
    std::cout << "not support sensor type: " << sensor_type << std::endl;
    return ret;
  }
  return 0;
}

static int GetDisAttr(SnsPlgCtrlInfo *ctrl_info,
     VIN_DIS_ATTR_S *pst_dis_attr) {
  return 0;
}

static int GetLdcAttr(SnsPlgCtrlInfo *ctrl_info,
     VIN_LDC_ATTR_S *pst_ldc_attr) {
  return 0;
}

SnsPlgModule gc4c33_plugin = {
  .module = "gc4c33_plugin",
  .get_sns_attr = GetSensorAttr,
  .get_mipi_attr = GetMipiAttr,
  .get_vin_dev_attr = GetVinDevAttr,
  .get_vin_dev_attr_ex = GetVinDevAttrEx,
  .get_vin_pipe_attr = GetVinPipeAttr,
  .get_dis_attr = GetDisAttr,
  .get_ldc_attr = GetLdcAttr,
};
