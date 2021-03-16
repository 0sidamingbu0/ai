/*
 * @Description: implement of data_type
 * @Author: ronghui.zhang@horizon.ai
 * @Date: 2020-11-02 17:49:26
 * @Copyright 2017~2019 Horizon Robotics, Inc.
 */

#include "LowPassFilterMethod/data_type.h"

namespace xstream {

int LowPassFilterParam::UpdateParameter(const JsonReaderPtr &reader) {
  LOGD << "MergeParam update config: " << this;
  if (reader) {
    reader_ = reader;
    #if 0
    SET_SNAPSHOT_METHOD_PARAM(reader_->GetRawJson(), Double, match_threshold);
    LOGD << "match_threshold: " << match_threshold;
    SET_SNAPSHOT_METHOD_PARAM(reader_->GetRawJson(), Int,
                              filtered_box_state_type);
    LOGD << "filtered_box_state_type: " << filtered_box_state_type;
    #endif
    return 0;
  } else {
    LOGE << "The reader is null";
    return -1;
  }
}

std::string LowPassFilterParam::Format() { return reader_->GetJson(); }

}  // namespace xstream
