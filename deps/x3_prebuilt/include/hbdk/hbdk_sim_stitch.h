//===-------- hbdk_sim_stitch.h - STITCH simulator interface ------*- C -*-===//
//
//                     The HBDK Simulator Infrastructure
//
// This file is subject to the terms and conditions defined in file
// 'LICENSE.txt', which is part of this source code package.
//
//===----------------------------------------------------------------===//
#ifndef HBDK3_HBDK_SIM_STITCH_H
#define HBDK3_HBDK_SIM_STITCH_H

#include "hbdk_config.h"
#include "hbdk_error.h"

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif  // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const uint8_t* src_y_address_0;
  const uint8_t* src_y_address_1;
  const uint8_t* src_uv_address_0;
  const uint8_t* src_uv_address_1;
  uint8_t* dst_y_address;
  uint8_t* dst_uv_address;
  const uint8_t* alpha_address;
  const uint8_t* beta_address;
} hbsim_stitch_address_message;

typedef struct hbsim_stitch_if {
  const uint32_t* config;  // Used to describe the parameters that generate roi
  uint32_t config_size;    // The size of config

  uint32_t stitch_roi_num;  // The number of roi

  hbsim_stitch_address_message address_message[64];
} hbsim_stitch_if;

HBDK_PUBLIC hbrt_error_t hbsimStitch(hbsim_stitch_if stitchIf);

#ifdef __cplusplus
}
#endif

#endif  // HBDK3_HBDK_SIM_STITCH_H
