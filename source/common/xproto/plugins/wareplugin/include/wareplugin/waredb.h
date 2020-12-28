/**
 * Copyright (c) 2020, Horizon Robotics, Inc.
 * All rights reserved.
 * @Author:
 * @Mail:
 * @Date: 2020-11-02 20:38:52
 * @Version: v0.0.1
 * @Last Modified by:
 * @Last Modified time: 2020-11-02 20:38:52
 */

#include <memory>
#include <string>
#include "horizon/vision_type/vision_type.h"
#include "horizon/vision_type/vision_type_util.h"
#include "ware_module/ware_module.h"
#include "aes/hobot_aes.h"

#ifndef _WAREPLUGIN_WAREDB_H_
#define _WAREPLUGIN_WAREDB_H_

namespace horizon {
namespace vision {
namespace xproto {
namespace wareplugin {

#define WARE_TABLE_BYTE_SIZE 128
typedef char *HorizonLibName;
typedef char *HorizonRecordId;
typedef char *HorizonImageUri;

struct HorizonLibRecordInfo {
  ~HorizonLibRecordInfo() {
    for (size_t i = 0; i < img_uri_list_size; i++) {
      if (img_uri_list && img_uri_list[i]) {
        std::free(img_uri_list[i]);
        img_uri_list[i] = nullptr;
      }
      if (feature_list) {
        HorizonVisionCleanCharArray(&feature_list[i]);
      }
    }
    img_uri_list_size = 0;
    if (img_uri_list) {
      std::free(img_uri_list);
    }
    img_uri_list = nullptr;
    if (feature_list) {
      std::free(feature_list);
    }
    feature_list = nullptr;
    if (id) {
      std::free(id);
    }
    id = nullptr;
    if (lib_name) {
      std::free(lib_name);
    }
    id = nullptr;
  }
  HorizonRecordId id = nullptr;
  HorizonLibName lib_name = nullptr;
  uint32_t img_uri_list_size = 0;
  HorizonImageUri *img_uri_list = nullptr;
  /// 特征向量列表,数量为img_uri_list_size
  HorizonVisionEncryptedFeature *feature_list = nullptr;
};

struct HorizonLibAuxInfo {
  /// \~Chinese 特征检索、比对时，特征的口罩屬性
  int32_t mask;
  /// \~Chinese 保留位
  int32_t reserved[3];
};

struct HorizonRecogInfo {
  // xwarehouse比对结果
  HorizonLibRecordInfo record_info;
  /// 是否匹配
  bool match = false;
  /// 特征距离
  float distance = 0;
  /// 特征相似度
  float similar = -1;
};

class DB {
 public:
  static std::shared_ptr<DB> Get(const std::string &table_path);
  static std::shared_ptr<DB> Get();
  ~DB();
  int Search(const std::string &lib_name, const std::string &model_version,
             HorizonVisionEncryptedFeature **feature_list,
             const uint32_t feature_list_num, HorizonRecogInfo *recog_info,
             HorizonLibAuxInfo*);
  int Search(const std::string &lib_name, const std::string &model_version,
             const char **feature_list, const uint32_t feature_list_num,
             uint32_t feature_size, HorizonRecogInfo *recog_info,
             HorizonLibAuxInfo*);

  int CreateTable(const std::string &lib_name,
                  const std::string &model_version);
  int DropTable(const std::string &lib_name, const std::string &model_version);
  int GetTableList(uint32_t num_limit, uint32_t start_index,
                   HorizonLibName **plib_list, uint32_t *lib_list_num);
  int GetTableList(uint32_t num_limit, uint32_t start_index,
                   HorizonLibName **plib_list, uint32_t *lib_list_num,
                   char ***model_version_list);
  int CreateRecordWithFeature(const std::string &lib_name,
                              const std::string &id,
                              const std::string &model_version,
                              const HorizonImageUri *img_uri_list,
                              HorizonVisionEncryptedFeature **feature_list,
                              uint32_t feature_list_num);
  int CreateRecordWithFeature(const std::string &lib_name,
                              const std::string &id,
                              const std::string &model_version,
                              const char **img_uri_list,
                              const char **feature_list,
                              uint32_t feature_list_num,
                              const uint32_t feature_size);
  int DropRecord(const std::string &lib_name, const std::string &id,
                 const std::string &model_version);
  int GetRecordIdList(const std::string &lib_name,
                      const std::string &model_version, uint32_t num_limit,
                      uint32_t start_index, HorizonRecordId **record_id_list,
                      uint32_t *record_id_num);
  int GetRecordInfoById(const std::string &lib_name, const std::string &id,
                        const std::string &model_version,
                        HorizonLibRecordInfo *record_info);
  int CompareFeatures(const HorizonVisionEncryptedFeature *feature1,
                      const HorizonVisionEncryptedFeature *feature2,
                      const float distance_threshold,
                      const float similar_threshold,
                      HorizonRecogInfo *recog_info,
                      HorizonLibAuxInfo*);
  int CompareFeatures(const char *feature1, const char *feature2,
                      const uint32_t feature_size, float *similarity,
                      HorizonLibAuxInfo*);

  static std::shared_ptr<DB> instance_;
  static std::string lib_name_;
  static std::string model_name_;
  void SetSimilarThres(float value) {
    similar_thres_ = value;
  }

 private:
  class SearchHelper {
   public:
    explicit SearchHelper(const std::string &lib_name,
                          const std::string &model_version,
                          const ware_param_t &search_param,
                          HorizonVisionEncryptedFeature **feature_list,
                          const uint32_t feature_list_num,
                          HorizonLibAuxInfo*);
    explicit SearchHelper(const std::string &lib_name,
                          const std::string &model_version,
                          const ware_param_t &search_param,
                          const char **feature_list,
                          const uint32_t feature_list_num,
                          const uint32_t feature_size,
                          HorizonLibAuxInfo*);
    int32_t Search(float similar_thres, HorizonRecogInfo *recog_info);
    ~SearchHelper();

   private:
    ware_table_t table_;
    ware_param_t param_;
    ware_search_t *search_result_ = nullptr;
  };

 private:
  explicit DB(const std::string &table_path);
  DB(const DB &db) = delete;
  DB &operator=(const DB &db) = delete;

  ware_param_t search_param_;
  float similar_thres_ = 0.74;
  std::string table_path_;
};

}  // namespace wareplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon

#endif  //  _WAREPLUGIN_WAREDB_H_


