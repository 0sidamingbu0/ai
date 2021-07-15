import sys
import os
sys.path.append("your/path/to/shared/libraries")
sys.path.append("your/path/to/xstream/xproto/packages")
sys.setdlopenflags(os.RTLD_LAZY)
import xstream    # noqa

FasterRCNNMethod = xstream.Method("FasterRCNNMethod").inputs(["image"])

# "CNNMethod" 对应MethodFactory中的CNNMethod的名字字符串。
CNNMethod = xstream.Method("CNNMethod")

# 声明inputs和outputs可以赋予输入输出参数以名字，
# 可以在生成的json中体现，让json更加可读
AgeGenderPreProcess = xstream.Method(
    "AgeGenderPreProcess").inputs(
        ["face_bbox_list", "image"]
    ).outputs(
        ["processed_image"]
    )

AgeGenderPostProcess = xstream.Method("AgeGenderPostProcess").inputs(
    ["cnn_out"]).outputs(["age_list", "gender_list"])
# agegender_cnn的配置定义。这里包括框架支撑的属性如thread_count，
# 也包括method自定义的配置如method_conf
age_gender_cnn_conf = {"threadCnt": 3, "methodConfig": "age_gender.json"}


def cnn_workflow(pre_method, post_method, cnn_conf, inputs, name=""):
    # name scope，在这个scope下，所有定义的符号的名字，都会加入scope的前缀
    with xstream.scope(name):
        # 调用 NativeMethod.__call__ 方法。由于scope的存在，
        # 全局的名字为 name + "/" + "pre"
        pre_out0 = pre_method(*inputs, unique_name="pre")
        cnn_out0 = CNNMethod(pre_out0, **cnn_conf, unique_name="cnn",
                             inputs=["processed_image"], outputs=["cnn_out"])
        outputs = post_method(cnn_out0, unique_name="post")
        return outputs


def main_workflow(image):
    face_box, lmk, pose = FasterRCNNMethod(
        image, outputs=["face_bbox_list", "lmk", "pose"],
        config_file="face_pose_lmk.json")
    age_list, gender_list = cnn_workflow(pre_method=AgeGenderPreProcess,
                                         post_method=AgeGenderPostProcess,
                                         cnn_conf=age_gender_cnn_conf,
                                         inputs=[face_box, image],
                                         name="age_gender_cnn")

    return age_list, gender_list


json_data = xstream.serialize(main_workflow)

print(json_data)
