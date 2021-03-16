/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2020 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#ifndef APP_INCLUDE_PLUGIN_COMMONGDCPLUGIN_MODULE_H_
#define APP_INCLUDE_PLUGIN_COMMONGDCPLUGIN_MODULE_H_
#include <memory>
#include <string>
#include <vector>
#include "commongdcplugin/gdcdata.h"

namespace horizon {
namespace vision {
namespace xproto {
namespace commongdcplugin {

class GdcModule {
 public:
  GdcModule() = delete;
  explicit GdcModule(std::shared_ptr<GdcConfig> &config);
  ~GdcModule() {}

  int Init();
  int DeInit();
  int Start();
  int Stop();
  int ConvertSrcInfo(void *src_buf, std::shared_ptr<SrcImageFrame> &src_img);
  void *CreateSrcAddrInfo();
  int Input(std::shared_ptr<PymImageFrame> &pym_img, int chn_id);
  int Output(int chn_id, int frame_id, void *gdc_buf);
  int FreeSrcImage(int chn_id, void *gdc_buf);

 private:
  int GetGdcData(const char *gdc_name, char **gdc_data, int *gdc_len);
  int CreateGrp(int grp_id, int grp_w, int grp_h, int grp_depth);
  int SetIpuChnAttr(int grp_id, int ipu_chn, int chn_w,
    int chn_h, int chn_depth);
  int SetPymAttr(int grp_id, int ipu_chn);
  int dump_nv12_file(std::string &filename, char *y_addr,
      char *uv_addr, uint32_t y_size, uint32_t uv_size);

 private:
  std::shared_ptr<GdcConfig> gdc_cfg_;
  int ipu_chn_;
  int pym_chn_;
  int gdc_input_w_;
  int gdc_input_h_;
  int gdc_output_w_;
  int gdc_output_h_;
  int gdc_frame_depth_;
};

}  // namespace commongdcplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  // APP_INCLUDE_PLUGIN_COMMONGDCPLUGIN_MODULE_H_
