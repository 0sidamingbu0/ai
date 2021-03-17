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

#include <cstring>
#include <mutex>
#include "horizon/vision_type/vision_type_util.h"
#include "horizon/vision_type/vision_error.h"
#include "wareplugin/waredb.h"
#include "hobotlog/hobotlog.hpp"

namespace horizon {
namespace vision {
namespace xproto {
namespace wareplugin {
#define CHECK_NULL                                 \
  if (lib_name.empty() || model_version.empty()) { \
    return -1;               \
  }

#define ADD_MASK_ATTR(attr, has_mask)    \
(attr |= has_mask << 7)

#define ADD_AUX_ATTR(attr, auxinfo)      \
  if (auxinfo != nullptr) {              \
    ADD_MASK_ATTR(attr, auxinfo->mask);  \
  }

#define IS_NOT_ENCRYPTED (1 << 6)

// #define WARE_OK 0
std::once_flag config_once;
std::shared_ptr<DB> DB::instance_;

std::string DB::lib_name_ = "aiexpressfaceid";  // NOLINT
std::string DB::model_name_ = "X2_1.6";  // NOLINT

/******************************************************************************
WARE_OK = 0,                        // ok
WARE_PARAM_ERR = 1000,              // 参数错误,字段为空，指针为空等
WARE_INIT_FAIL = 1001,              // 初始化失败
WARE_CREATE_TABLE_FAIL = 1002,      // 插入set信息失败
WARE_DELETE_TABLE_FAIL = 1003,      // 删除set信息失败
WARE_UPDATE_THRESHOLD_FAIL = 1004,  // 更新阈值失败
WARE_LIST_TABLE_FAIL = 1005,        // list set失败
WARE_LOAD_TABLE_FAIL = 1006,        // 加载数据库中的set信息失败
WARE_SOURCETYPE_NOT_MATCH = 1007,   // 类型不匹配
WARE_TABLE_INIT_FAIL = 1010,        // 分库初始化失败
WARE_TABLE_EXIST = 1011,            // 分库已经存在
WARE_TABLE_NOT_EXIST = 1012,        // 分库不存在
WARE_FEATURE_SIZE_NOT_MATCH = 1013, // 特征向量大小不匹配
WARE_ADD_RECORD_FAIL = 1014,        // 新增记录失败
WARE_DELETE_RECORD_FAIL = 1015,     // 删除记录失败
WARE_UPDATE_RECORD_FAIL = 1016,     // 更新记录失败
WARE_ADD_FEATURE_FAIL = 1017,       // 新增特征失败
WARE_DELETE_FEATURE_FAIL = 1018,    // 删除特征失败
WARE_UPDATE_FEATURE_FAIL = 1019,    // 更新特征失败
WARE_ID_EXIST = 1020,               // id已经存在
WARE_ID_NOT_EXIST = 1021,           // id不存在
WARE_DECRYPT_FAIL = 1022,           // 特征值解密失败
WARE_OVER_LIMIT = 1023,             // 超过库容
WARE_GET_LIMIT_FAIL = 1024,         // 获取库容大小失败
WARE_GET_RECORD_FAIL = 1025,        // 获取记录失败
WARE_NOT_INIT = 1026,               // ware_module未初始化
*******************************************************************************/
namespace {
std::string GetWareModuleErrorDetail(int int_code) {
  auto code = (ware_error_t)(int_code);
  switch (code) {
//    case WARE_OK:
//      return "WARE_OK";
    case WARE_PARAM_ERR:return "WARE_PARAM_ERR";
    case WARE_INIT_FAIL:return "WARE_INIT_FAIL";
    case WARE_CREATE_TABLE_FAIL:return "WARE_CREATE_TABLE_FAIL";
    case WARE_DELETE_TABLE_FAIL:return "WARE_DELETE_TABLE_FAIL";
    case WARE_UPDATE_THRESHOLD_FAIL:return "WARE_UPDATE_THRESHOLD_FAIL";
    case WARE_LIST_TABLE_FAIL:return "WARE_LIST_TABLE_FAIL";
    case WARE_LOAD_TABLE_FAIL:return "WARE_LOAD_TABLE_FAIL";
    case WARE_SOURCETYPE_NOT_MATCH:return "WARE_SOURCETYPE_NOT_MATCH";
    case WARE_TABLE_INIT_FAIL:return "WARE_TABLE_INIT_FAIL";
    case WARE_TABLE_EXIST:return "WARE_TABLE_EXIST";
    case WARE_TABLE_NOT_EXIST:return "WARE_TABLE_NOT_EXIST";
    case WARE_FEATURE_SIZE_NOT_MATCH:return "WARE_FEATURE_SIZE_NOT_MATCH";
    case WARE_ADD_RECORD_FAIL:return "WARE_ADD_RECORD_FAIL";
    case WARE_DELETE_RECORD_FAIL:return "WARE_DELETE_RECORD_FAIL";
    case WARE_UPDATE_RECORD_FAIL:return "WARE_UPDATE_RECORD_FAIL";
    case WARE_ADD_FEATURE_FAIL:return "WARE_ADD_FEATURE_FAIL";
    case WARE_DELETE_FEATURE_FAIL:return "WARE_DELETE_FEATURE_FAIL";
    case WARE_UPDATE_FEATURE_FAIL:return "WARE_UPDATE_FEATURE_FAIL";
    case WARE_ID_EXIST:return "WARE_ID_EXIST";
    case WARE_ID_NOT_EXIST:return "WARE_ID_NOT_EXIST";
    case WARE_DECRYPT_FAIL:return "WARE_DECRYPT_FAIL";
    case WARE_OVER_LIMIT:return "WARE_OVER_LIMIT";
    case WARE_GET_LIMIT_FAIL:return "WARE_GET_LIMIT_FAIL";
    case WARE_GET_RECORD_FAIL:return "WARE_GET_RECORD_FAIL";
    case WARE_NOT_INIT:return "WARE_NOT_INIT";
    default:return "NOT_COMMON_ERROR";
  }
}

ware_table_t GetTable(const std::string &lib_name,
                      const std::string &model_version) {
  ware_table_t table;
  memset(table.table_name, 0, ID_MAX_LENGTH);
  memset(table.model_version, 0, VERSION_MAX_LENGTH);
  std::snprintf(table.table_name, ID_MAX_LENGTH, "%s", lib_name.c_str());
  std::snprintf(table.model_version, VERSION_MAX_LENGTH, "%s",
                model_version.c_str());
  return table;
}
}  //  namespace

DB::DB(const std::string &table_path) {
  table_path_ = table_path;
  ware_module_set_debug_level(WARE_DEBUG_LOW);
  ware_module_init(const_cast<char *>(table_path_.c_str()), WARE_SQLITE);
}

DB::~DB() { ware_module_deinit(); }

std::shared_ptr<DB> DB::Get(const std::string &table_path) {
  if (!instance_) {
    HOBOT_CHECK(!table_path.empty());
    std::call_once(config_once, [&]() {
      instance_ = std::shared_ptr<DB>(new DB(table_path));
    });
  }
  return instance_;
}

std::shared_ptr<DB> DB::Get() {
  if (instance_ == nullptr) {
    LOGE << "db has not been initialized yet.";
    return nullptr;
  }
  return instance_;
}

int DB::CreateTable(const std::string &lib_name,
                    const std::string &model_version) {
  CHECK_NULL
  auto table = GetTable(lib_name, model_version);
  auto ret =
      ware_module_create_table(
          const_cast<char *>(table_path_.c_str()),
          WARE_SQLITE, &table, WARE_TABLE_BYTE_SIZE, WARE_CHECK_NULL);
  if (WARE_TABLE_EXIST == ret) {
//    LOGI << "Failed to call ware_module_create_table, error code is " << ret
//         << " error msg is " << GetWareModuleErrorDetail(ret);
    LOGI << GetWareModuleErrorDetail(ret);
    return 0;
  } else if (WARE_OK != ret) {
    LOGE << "Failed to call ware_module_create_table, error code is " << ret
         << " error msg is " << GetWareModuleErrorDetail(ret);
    return ret;
  } else {
    return 0;
  }
}

int DB::DropTable(const std::string &lib_name,
                  const std::string &model_version) {
  CHECK_NULL
  auto table = GetTable(lib_name, model_version);
  auto ret = ware_module_destroy_table(&table);
  if (WARE_OK != ret) {
    return ret;
  }
  return 0;
}

int DB::GetTableList(uint32_t num_limit, uint32_t start_index,
                     HorizonLibName **plib_list, uint32_t *lib_list_num) {
  if (nullptr == plib_list || nullptr == lib_list_num) {
    return -1;
  }
  ware_table_list_t *table_list = nullptr;
  auto ret = ware_module_list_table(&table_list);
  if (WARE_OK != ret) {
    LOGE << "Failed to call ware_module_list_table, error code is " << ret;
    return ret;
  }

  uint32_t total_list_num = static_cast<uint32_t>(table_list->num);
  if (total_list_num == 0 || start_index > total_list_num - 1) {
    *lib_list_num = 0;
  } else {
    *lib_list_num = total_list_num - start_index;
  }

  if (0 != num_limit) {
    *lib_list_num = std::min(*lib_list_num, num_limit);
  }
  *plib_list = reinterpret_cast<HorizonLibName *>(
      std::calloc(*lib_list_num, sizeof(HorizonLibName)));
  for (uint32_t i = 0; i < *lib_list_num; ++i) {
    (*plib_list)[i] =
        HorizonVisionStrDup(table_list->table_info[i + start_index].table_name);
  }
  ware_module_release_table_list(&table_list);

  return 0;
}

int DB::GetTableList(uint32_t num_limit, uint32_t start_index,
                     HorizonLibName **plib_list, uint32_t *lib_list_num,
                     char ***model_version_list) {
  if (nullptr == plib_list || nullptr == lib_list_num ||
      nullptr == model_version_list) {
    return -1;
  }
  ware_table_list_t *table_list = nullptr;
  auto ret = ware_module_list_table(&table_list);
  if (WARE_OK != ret) {
    LOGE << "Failed to call ware_module_list_table, error code is " << ret;
    return ret;
  }

  uint32_t total_list_num = static_cast<uint32_t>(table_list->num);
  if (total_list_num == 0 || start_index > total_list_num - 1) {
    *lib_list_num = 0;
  } else {
    *lib_list_num = total_list_num - start_index;
  }

  if (0 != num_limit) {
    *lib_list_num = std::min(*lib_list_num, num_limit);
  }
  *plib_list = reinterpret_cast<HorizonLibName *>(
      std::calloc(*lib_list_num, sizeof(HorizonLibName)));
  *model_version_list =
      reinterpret_cast<char **>(std::calloc(*lib_list_num, sizeof(char **)));
  for (uint32_t i = 0; i < *lib_list_num; ++i) {
    (*plib_list)[i] =
        HorizonVisionStrDup(table_list->table_info[i + start_index].table_name);
    (*model_version_list)[i] =
        HorizonVisionStrDup(
            table_list->table_info[i + start_index].model_version);
  }
  ware_module_release_table_list(&table_list);

  return 0;
}

ware_feature_t *ConstructWareFeature(
    const char *feature,
    const uint32_t feature_size,
    HorizonLibAuxInfo *attr_info = nullptr,
    ware_feature_t *features = nullptr
) {
  ware_feature_t *ware_feature =
      (features == nullptr) ?
      reinterpret_cast<ware_feature_t *>(
          std::calloc(1, sizeof(ware_feature_t))) : features;
  ware_feature->attr = 1 | IS_NOT_ENCRYPTED;
  ware_feature->size = feature_size;
  ware_feature->feature =
      reinterpret_cast<float *>(std::calloc(feature_size, sizeof(float)));
  ADD_AUX_ATTR(ware_feature->attr, attr_info);
  memcpy(ware_feature->feature, feature, sizeof(float) * feature_size);
  return ware_feature;
}

int DB::CreateRecordWithFeature(const std::string &lib_name,
                                const std::string &id,
                                const std::string &model_version,
                                const HorizonImageUri *img_uri_list,
                                HorizonVisionEncryptedFeature **feature_list,
                                uint32_t feature_list_num) {
  CHECK_NULL
  if (feature_list == nullptr || feature_list_num == 0) {
    LOGE << "feature_list is null";
    return -1;
  }

  ware_record_t record;
  memset(record.record_id, 0, ID_MAX_LENGTH);
//  strcpy(record.record_id, id.c_str());
  snprintf(record.record_id, ID_MAX_LENGTH, "%s", id.c_str());

  record.num = feature_list_num;
//  ware_feature_t features[feature_list_num];
//  record.features = features;
  ware_feature_t *features_ptr = new ware_feature_t[feature_list_num];
  record.features = features_ptr;
  for (uint32_t i = 0; i < feature_list_num; ++i) {
    size_t feature_size = feature_list[i]->num / sizeof(float);
    ConstructWareFeature(feature_list[i]->values,
                         feature_size, nullptr, &record.features[i]);
//    std::strcpy(record.features[i].img_uri, img_uri_list[i]);
    snprintf(record.features[i].img_uri, URI_MAX_LENGTH, "%s", img_uri_list[i]);
  }
  auto table = GetTable(lib_name, model_version);
  auto ret = ware_module_add_record(&table, &record);
  for (uint32_t i = 0; i < feature_list_num; ++i) {
    std::free(record.features[i].feature);
  }
  if (nullptr != features_ptr) {
    delete[] features_ptr;
    features_ptr = nullptr;
  }
  if (WARE_OK != ret) {
    LOGE << "Failed to CreateRecord " << id << " into " << lib_name << ":"
         << model_version << ", ware_module_add_record return val is " << ret
         << " error msg is " << GetWareModuleErrorDetail(ret);
    return ret;
  }
  LOGI << "Succeed to CreateRecord into " << lib_name << ":" << model_version
       << "id: " << id;
  return 0;
}

int DB::CreateRecordWithFeature(const std::string &lib_name,
                                const std::string &id,
                                const std::string &model_version,
                                const char **img_uri_list,
                                const char **feature_list,
                                uint32_t feature_list_num,
                                const uint32_t feature_size) {
  CHECK_NULL
  if (nullptr == feature_list || 0 == feature_list_num ||
      nullptr == img_uri_list) {
    LOGE << "feature_list is null";
    return -1;
  }

  ware_record_t record;
  memset(record.record_id, 0, ID_MAX_LENGTH);
//  strcpy(record.record_id, id.c_str());
  snprintf(record.record_id, ID_MAX_LENGTH, "%s", id.c_str());

  record.num = feature_list_num;
//  ware_feature_t features[feature_list_num];
//  record.features = features;
  ware_feature_t *features_ptr = new ware_feature_t[feature_list_num];
  record.features = features_ptr;
  size_t feature_size_var = feature_size / sizeof(float);
  for (uint32_t i = 0; i < feature_list_num; ++i) {
    ConstructWareFeature(feature_list[i],
                         feature_size_var, nullptr, &record.features[i]);
//    std::strcpy(record.features[i].img_uri, img_uri_list[i]);
    snprintf(record.features[i].img_uri, URI_MAX_LENGTH, "%s", img_uri_list[i]);
  }

  auto table = GetTable(lib_name, model_version);
  auto ret = ware_module_add_record(&table, &record);
  for (uint32_t i = 0; i < feature_list_num; ++i) {
    std::free(record.features[i].feature);
  }
  if (nullptr != features_ptr) {
    delete[] features_ptr;
    features_ptr = nullptr;
  }
  if (WARE_OK != ret) {
    LOGE << "Failed to CreateRecord " << id << " into " << lib_name << ":"
         << model_version << ", ware_module_add_record return val is " << ret
         << " error msg is " << GetWareModuleErrorDetail(ret);
    return ret;
  }
  LOGI << "Succeed to CreateRecord into " << lib_name << ":" << model_version
       << ":" << id;
  return 0;
}

int DB::DropRecord(const std::string &lib_name, const std::string &id,
                   const std::string &model_version) {
  CHECK_NULL
  auto table = GetTable(lib_name, model_version);
  auto ret = ware_module_delete_record(&table, const_cast<char *>(id.c_str()));
  if (WARE_OK != ret) {
    LOGE << "Failed to DropRecord " << id << " into " << lib_name << ":"
         << model_version << ", ware_module_add_record return val is " << ret
         << " error msg is " << GetWareModuleErrorDetail(ret);
    return ret;
  }
  return 0;
}

int DB::GetRecordIdList(const std::string &lib_name,
                        const std::string &model_version, uint32_t num_limit,
                        uint32_t start_index,
                        HorizonRecordId **record_id_list,
                        uint32_t *record_id_num) {
  if (nullptr == record_id_list || nullptr == record_id_num) {
    return -1;
  }
  CHECK_NULL
  auto table = GetTable(lib_name, model_version);
  ware_record_list_t *records = nullptr;
  auto ret = ware_module_list_record(&table, &records);
  if (WARE_OK != ret) {
    LOGE << "Failed to call ware_module_list_record, error code is " << ret;
    return ret;
  }
  uint32_t total_id_num = static_cast<uint32_t>(records->num);
  LOGD << "total id num: " << total_id_num;
  if (total_id_num == 0 || start_index > total_id_num - 1) {
    *record_id_num = 0;
  } else {
    *record_id_num = total_id_num - start_index;
  }
  if (0 != num_limit) {
    *record_id_num = std::min(*record_id_num, num_limit);
  }
  LOGD << "record id num: " << *record_id_num;
  if (0 == *record_id_num) {
    return 0;
  }
  *record_id_list = reinterpret_cast<HorizonRecordId *>(
      std::calloc(*record_id_num, sizeof(HorizonRecordId)));
  for (uint32_t i = 0; i < *record_id_num; ++i) {
    (*record_id_list)[i] =
        HorizonVisionStrDup(records->record[i + start_index].record_id);
  }
  ware_module_release_record_list(&records);
  return 0;
}

int DB::GetRecordInfoById(const std::string &lib_name, const std::string &id,
                          const std::string &model_version,
                          HorizonLibRecordInfo *record_info) {
  CHECK_NULL
  if (nullptr == record_info) {
    return -1;
  }
  int ret = 0;
  auto table = GetTable(lib_name, model_version);
  ware_ftr_list_t *features = nullptr;
  ret = ware_module_list_feature(&table, const_cast<char *>(id.c_str()),
                                 &features);
  if (WARE_ID_NOT_EXIST == ret) {
    ware_module_release_feature_list(&features);
    LOGI << "ID not exist.";
    return ret;
  }
  if (WARE_OK != ret) {
    ware_module_release_feature_list(&features);
    LOGE << "Failed to call ware_module_list_feature, error code is " << ret;
    return ret;
  }
  HOBOT_CHECK(0 == std::strcmp(features->ware_record.record_id, id.c_str()));
  record_info->id = HorizonVisionStrDup(id.c_str());
  record_info->lib_name = HorizonVisionStrDup(lib_name.c_str());
  int uri_list_size = features->ware_record.num;
  record_info->img_uri_list_size = uri_list_size;
  record_info->img_uri_list = reinterpret_cast<HorizonImageUri *>(
      std::calloc(uri_list_size, sizeof(HorizonImageUri)));
  record_info->feature_list =
      reinterpret_cast<HorizonVisionEncryptedFeature *>(
          std::calloc(uri_list_size, sizeof(HorizonVisionEncryptedFeature)));
  for (int j = 0; j < uri_list_size; ++j) {
    record_info->img_uri_list[j] =
        HorizonVisionStrDup(features->ware_record.features[j].img_uri);
    int type_size = 1;
    if (!(features->ware_record.features[j].attr & 0x01)) {
      type_size = sizeof(int);
    } else if ((features->ware_record.features[j].attr & 0x01)) {
      type_size = sizeof(float);
    }
    record_info->feature_list[j].num =
        features->ware_record.features[j].size * type_size;
    record_info->feature_list[j].values =
        HorizonVisionMemDup(reinterpret_cast<char *>(
                                features->ware_record.features[j].feature),
                            features->ware_record.features[j].size * type_size);
  }
  ret = ware_module_release_feature_list(&features);
  if (WARE_OK != ret) {
    LOGE << "Failed to call ware_module_release_feature_list, "
            "error code is "
         << ret;
    return ret;
  }
  return 0;
}

int DB::Search(const std::string &lib_name, const std::string &model_version,
               HorizonVisionEncryptedFeature **feature_list,
               const uint32_t feature_list_num,
               HorizonRecogInfo *recog_info,
               HorizonLibAuxInfo *attr_info) {
  CHECK_NULL
  HOBOT_CHECK(similar_thres_ >= 0) << "similar thres is null!";
  if (feature_list == nullptr || recog_info == nullptr ||
      feature_list_num == 0) {
    return -1;
  }
  SearchHelper search_helper(lib_name, model_version, search_param_,
                             feature_list, feature_list_num, attr_info);
  auto ret = search_helper.Search(similar_thres_, recog_info);
  if (WARE_OK != ret) {
    LOGE << "Failed to Search feature in " << lib_name << ":" << model_version
         << ", error code is " << ret << ", error msg is "
         << GetWareModuleErrorDetail(ret);
    return ret;
  }
  LOGI << "Succeed to Search feature in  " << lib_name << ":" << model_version;
  return 0;
}

int DB::Search(const std::string &lib_name, const std::string &model_version,
               const char **feature_list, const uint32_t feature_list_num,
               uint32_t feature_size, HorizonRecogInfo *recog_info,
               HorizonLibAuxInfo *attr_info) {
  CHECK_NULL
  HOBOT_CHECK(similar_thres_ >= 0) << "similar thres is null!";
  if (feature_list == nullptr || recog_info == nullptr ||
      feature_list_num == 0) {
    return -1;
  }
  SearchHelper search_helper(lib_name, model_version, search_param_,
                             feature_list, feature_list_num, feature_size,
                             attr_info);
  auto ret = search_helper.Search(similar_thres_, recog_info);
  if (WARE_OK != ret) {
    LOGE << "Failed to Search feature in " << lib_name << ":" << model_version
         << ", error code is " << ret << ", error msg is "
         << GetWareModuleErrorDetail(ret);
    return ret;
  }
  LOGI << "Succeed to Search feature in " << lib_name << ":" << model_version;
  return 0;
}

ware_record_t ConstructWareRecord(
    const HorizonVisionEncryptedFeature *feature,
    HorizonLibAuxInfo *attr_info) {
  HOBOT_CHECK(feature) << "empty feature!";
  ware_record_t ware_record;
  size_t feature_size = feature->num / sizeof(float);
  auto &ware_feature = ware_record.features;
  ware_feature = ConstructWareFeature(
      feature->values, feature_size, attr_info);
  ware_record.num = 1;
  return ware_record;
}

int DB::CompareFeatures(const HorizonVisionEncryptedFeature *feature1,
                        const HorizonVisionEncryptedFeature *feature2,
                        const float distance_threshold,
                        const float similar_threshold,
                        HorizonRecogInfo *recog_info,
                        HorizonLibAuxInfo *attr_info) {
  ware_record_t ware_record1, ware_record2;
  if (!recog_info) {
    return -1;
  } else {
    if (attr_info == nullptr) {
      ware_record1 = ConstructWareRecord(feature1, nullptr);
      ware_record2 = ConstructWareRecord(feature2, nullptr);
    } else {
      ware_record1 = ConstructWareRecord(feature1, attr_info);
      ware_record2 = ConstructWareRecord(feature2, attr_info + 1);
    }
    ware_compare_t ware_compare;

    int ret = ware_module_compare_record(&ware_record1, &ware_record2,
                                         distance_threshold, similar_threshold,
                                         &ware_compare);
    std::free(ware_record1.features->feature);
    std::free(ware_record2.features->feature);
    std::free(ware_record1.features);
    std::free(ware_record2.features);
    if (ret != WARE_OK) {
      LOGE << "Failed to call ware_module_compare_record, error code is " << ret
           << " error msg is " << GetWareModuleErrorDetail(ret);
      return ret;
    }
    recog_info->match = ware_compare.match;
    recog_info->distance = ware_compare.distance;
    recog_info->similar = ware_compare.similar;
    return 0;
  }
}

ware_record_t ConstructWareRecord(const char *feature,
                                  const uint32_t feature_size,
                                  HorizonLibAuxInfo *attr_info) {
  HOBOT_CHECK(feature) << "empty feature!";
  ware_record_t ware_record;
  auto &ware_feature = ware_record.features;
  size_t feature_size_var = feature_size / sizeof(float);
  ware_feature = ConstructWareFeature(feature, feature_size_var, attr_info);
  ware_record.num = 1;
  return ware_record;
}

int DB::CompareFeatures(const char *feature1, const char *feature2,
                        const uint32_t feature_size, float *similarity,
                        HorizonLibAuxInfo *attr_info) {
  ware_record_t ware_record1, ware_record2;
  if (nullptr == similarity) {
    return -1;
  }
  if (attr_info == nullptr) {
    ware_record1 = ConstructWareRecord(feature1, feature_size, nullptr);
    ware_record2 = ConstructWareRecord(feature2, feature_size, nullptr);
  } else {
    ware_record1 = ConstructWareRecord(feature1, feature_size, attr_info);
    ware_record2 = ConstructWareRecord(feature2, feature_size, attr_info + 1);
  }

  ware_compare_t ware_compare;
  int ret = ware_module_compare_record(&ware_record1, &ware_record2,
                                       0, 0, &ware_compare);
  std::free(ware_record1.features->feature);
  std::free(ware_record2.features->feature);
  std::free(ware_record1.features);
  std::free(ware_record2.features);
  if (ret != WARE_OK) {
    LOGE << "Failed to call ware_module_compare_record, error code is " << ret
         << " error msg is " << GetWareModuleErrorDetail(ret);
    return ret;
  }
//  similarity =
//    reinterpret_cast<float *>(std::calloc(1, sizeof(float)));
  *similarity = ware_compare.similar;
  return 0;
}

DB::SearchHelper::SearchHelper(
    const std::string &lib_name, const std::string &model_version,
    const ware_param_t &search_param,
    HorizonVisionEncryptedFeature **feature_list,
    const uint32_t feature_list_num, HorizonLibAuxInfo *attr_info) {
  table_ = GetTable(lib_name, model_version);
  param_ = search_param;
  param_.num = feature_list_num;
  param_.features = new ware_feature_t[param_.num];
  for (uint32_t i = 0; i < feature_list_num; ++i) {
    size_t feature_size = feature_list[i]->num / sizeof(float);
    if (attr_info != nullptr) {
      ConstructWareFeature(
          feature_list[i]->values, feature_size,
          attr_info + i, &param_.features[i]);
    } else {
      ConstructWareFeature(
          feature_list[i]->values, feature_size,
          nullptr, &param_.features[i]);
    }
  }
  search_result_ = ware_module_alloc_search();
}

DB::SearchHelper::SearchHelper(
    const std::string &lib_name, const std::string &model_version,
    const ware_param_t &search_param, const char **feature_list,
    const uint32_t feature_list_num, const uint32_t feature_size,
    HorizonLibAuxInfo *attr_info) {
  table_ = GetTable(lib_name, model_version);
  param_ = search_param;
  param_.num = feature_list_num;
  param_.features = new ware_feature_t[param_.num];
  size_t feature_size_var = feature_size / sizeof(float);
  for (uint32_t i = 0; i < feature_list_num; ++i) {
    if (attr_info != nullptr) {
      ConstructWareFeature(
          feature_list[i],
          feature_size_var, attr_info + i, &param_.features[i]);
    } else {
      ConstructWareFeature(feature_list[i],
                           feature_size_var, nullptr, &param_.features[i]);
    }
  }
  search_result_ = ware_module_alloc_search();
}

int32_t DB::SearchHelper::Search(float similar_thres,
                                 HorizonRecogInfo *recog_info) {
  auto ret = ware_module_search_record(&table_, &param_, search_result_);
  if (WARE_OK == ret) {
    HOBOT_CHECK(search_result_ != nullptr);
    if (search_result_->num > 0) {
      LOGD << "has match result, match num = " << search_result_->num;
      auto top1 = search_result_->id_score[0];
      LOGI << "find match target, id is " << top1.id
           << ", similar: " << top1.similar << ", distance: " << top1.distance;
      recog_info->match = top1.similar > similar_thres;
      recog_info->distance = top1.distance;
//      std::strcpy(recog_info->record_info.id, top1.id);
      recog_info->record_info.lib_name = HorizonVisionStrDup(table_.table_name);
      recog_info->record_info.id = HorizonVisionStrDup(top1.id);
      recog_info->similar = top1.similar;
      ware_ftr_list_t *ftr_list = nullptr;
      ret = ware_module_list_feature(&table_, top1.id, &ftr_list);
      // after the record is matched, there is a danger that the record
      // will be deleted in other place
      if (ret == WARE_OK && ftr_list->ware_record.num > 0) {
        int record_size = ftr_list->ware_record.num;
        recog_info->record_info.img_uri_list_size = record_size;
        recog_info->record_info.img_uri_list =
            reinterpret_cast<HorizonImageUri *>(
                std::calloc(record_size, sizeof(HorizonImageUri)));

        recog_info->record_info.feature_list =
            reinterpret_cast<HorizonVisionEncryptedFeature *>(std::calloc(
                record_size, sizeof(HorizonVisionEncryptedFeature)));
        auto feature_list_var = recog_info->record_info.feature_list;
        for (int i = 0; i < record_size; ++i) {
          recog_info->record_info.img_uri_list[i] =
              HorizonVisionStrDup(ftr_list->ware_record.features[i].img_uri);
          int type_size = 1;
          if (!(ftr_list->ware_record.features[i].attr & 0x01)) {
            type_size = sizeof(int);
          } else if (ftr_list->ware_record.features[i].attr) {
            type_size = sizeof(float);
          }
          feature_list_var[i].num =
              ftr_list->ware_record.features[i].size * type_size;
          feature_list_var[i].values = HorizonVisionMemDup(
              reinterpret_cast<char *>
              (ftr_list->ware_record.features[i].feature),
              ftr_list->ware_record.features[i].size * type_size);
        }
        ware_module_release_feature_list(&ftr_list);
      }
    }
  }
  return ret;
}

DB::SearchHelper::~SearchHelper() {
  for (int i = 0; i < param_.num; ++i) {
    std::free(param_.features[i].feature);
  }
  delete[] param_.features;
  ware_module_release_record_search(&search_result_);
}

}  // namespace wareplugin
}  // namespace xproto
}  // namespace vision
}  // namespace horizon
