command_x3 = "echo 1000000000 > /sys/class/devfreq/devfreq1/userspace/set_freq;\
    echo 105000 >/sys/devices/virtual/thermal/thermal_zone0/trip_point_1_temp; \
    echo performance > \
    /sys/devices/system/cpu/cpufreq/policy0/scaling_governor;\
    echo 0 > /sys/class/misc/ion/cma_carveout_size; "

command_imx327 = "echo 1 > /sys/class/vps/mipi_host1/param/stop_check_instart \
    cp -rf body_solution/configs/box_filter_config_2M.json \
    body_solution/configs/box_filter_config.json; \
    cp -rf body_solution/configs/multitask_config_2M.json \
    body_solution/configs/multitask_config.json; \
    cp -rf body_solution/configs/gesture_multitask_2M.json \
    body_solution/configs/gesture_multitask.json; \
    cp -rf body_solution/configs/multitask_with_hand_2M.json \
    body_solution/configs/multitask_with_hand.json; \
    cp -rf body_solution/configs/segmentation_multitask_2M.json \
    body_solution/configs/segmentation_multitask.json; \
    cp -rf face_solution/configs/face_pose_lmk_2M.json \
    face_solution/configs/face_pose_lmk.json"

command_os8a10 = \
    "echo start > /sys/devices/virtual/graphics/iar_cdev/iar_test_attr; \
    echo start > /sys/devices/virtual/graphics/iar_cdev/iar_test_attr; \
    /etc/init.d/usb-gadget.sh start uvc-hid; \
    echo 0 > /proc/sys/kernel/printk; \
    echo 1 > /sys/class/vps/mipi_host1/param/stop_check_instart; \
    echo 0xc0120000 > /sys/bus/platform/drivers/ddr_monitor/axibus_ctrl/all; \
    echo 0x03120000 > /sys/bus/platform/drivers/ddr_monitor/read_qos_ctrl/all; \
    echo 0x03120000 > /sys/bus/platform/drivers/ddr_monitor/write_qos_ctrl/all;\
    cp -rf body_solution/configs/box_filter_config_8M.json \
    body_solution/configs/box_filter_config.json; \
    cp -rf body_solution/configs/multitask_config_8M.json \
    body_solution/configs/multitask_config.json; \
    cp -rf body_solution/configs/gesture_multitask_8M.json \
    body_solution/configs/gesture_multitask.json; \
    cp -rf body_solution/configs/multitask_with_hand_8M.json \
    body_solution/configs/multitask_with_hand.json; \
    cp -rf body_solution/configs/segmentation_multitask_8M.json \
    body_solution/configs/segmentation_multitask.json; \
    cp -rf face_solution/configs/face_pose_lmk_8M.json \
    face_solution/configs/face_pose_lmk.json"

command_os8a10_1080p = "cp -rf body_solution/configs/box_filter_config_2M.json \
    body_solution/configs/box_filter_config.json; \
    cp -rf body_solution/configs/multitask_config_2M.json \
    body_solution/configs/multitask_config.json; \
    cp -rf body_solution/configs/gesture_multitask_2M.json \
    body_solution/configs/gesture_multitask.json; \
    cp -rf body_solution/configs/multitask_with_hand_2M.json \
    body_solution/configs/multitask_with_hand.json; \
    cp -rf body_solution/configs/segmentation_multitask_2M.json \
    body_solution/configs/segmentation_multitask.json; \
    cp -rf face_solution/configs/face_pose_lmk_2M.json \
    face_solution/configs/face_pose_lmk.json"

command_s5kgm = "cp -rf body_solution/configs/box_filter_config_12M.json \
    body_solution/configs/box_filter_config.json; \
    cp -rf body_solution/configs/multitask_config_12M.json \
    body_solution/configs/multitask_config.json; \
    cp -rf body_solution/configs/gesture_multitask_12M.json \
    body_solution/configs/gesture_multitask.json; \
    cp -rf body_solution/configs/multitask_with_hand_12M.json \
    body_solution/configs/multitask_with_hand.json; \
    cp -rf body_solution/configs/segmentation_multitask_12M.json \
    body_solution/configs/segmentation_multitask.json"

command_s5kgm_2160p = \
    "echo start > /sys/devices/virtual/graphics/iar_cdev/iar_test_attr; \
    service adbd stop; \
    /etc/init.d/usb-gadget.sh start uvc-hid; \
    echo 0 > /proc/sys/kernel/printk; \
    echo 1 > /sys/class/vps/mipi_host1/param/stop_check_instart; \
    echo 0xc0120000 > /sys/bus/platform/drivers/ddr_monitor/axibus_ctrl/all; \
    echo 0x03120000 > /sys/bus/platform/drivers/ddr_monitor/read_qos_ctrl/all; \
    echo 0x03120000 > /sys/bus/platform/drivers/ddr_monitor/write_qos_ctrl/all;\
    cp -rf body_solution/configs/box_filter_config_8M.json \
    body_solution/configs/box_filter_config.json; \
    cp -rf body_solution/configs/multitask_config_8M.json \
    body_solution/configs/multitask_config.json; \
    cp -rf body_solution/configs/gesture_multitask_8M.json \
    body_solution/configs/gesture_multitask.json; \
    cp -rf body_solution/configs/multitask_with_hand_8M.json \
    body_solution/configs/multitask_with_hand.json; \
    cp -rf body_solution/configs/segmentation_multitask_8M.json \
    body_solution/configs/segmentation_multitask.json; \
    cp -rf face_solution/configs/face_pose_lmk_8M.json \
    face_solution/configs/face_pose_lmk.json"

command_hg = "service adbd stop; \
    /etc/init.d/usb-gadget.sh start uvc-hid; \
    cp -rf body_solution/configs/box_filter_config_2M.json \
    body_solution/configs/box_filter_config.json; \
    cp -rf body_solution/configs/multitask_config_2M.json \
    body_solution/configs/multitask_config.json; \
    cp -rf body_solution/configs/gesture_multitask_2M.json \
    body_solution/configs/gesture_multitask.json; \
    cp -rf body_solution/configs/multitask_with_hand_2M.json \
    body_solution/configs/multitask_with_hand.json; \
    cp -rf body_solution/configs/segmentation_multitask_2M.json \
    body_solution/configs/segmentation_multitask.json; \
    cp -rf face_solution/configs/face_pose_lmk_2M.json \
    face_solution/configs/face_pose_lmk.json"

command_usb_cam = "service adbd stop; \
    cp -rf body_solution/configs/box_filter_config_2M.json \
    body_solution/configs/box_filter_config.json; \
    cp -rf body_solution/configs/multitask_config_2M.json \
    body_solution/configs/multitask_config.json; \
    cp -rf body_solution/configs/gesture_multitask_2M.json \
    body_solution/configs/gesture_multitask.json; \
    cp -rf body_solution/configs/multitask_with_hand_2M.json \
    body_solution/configs/multitask_with_hand.json; \
    cp -rf body_solution/configs/segmentation_multitask_2M.json \
    body_solution/configs/segmentation_multitask.json; \
    cp -rf face_solution/configs/face_pose_lmk_2M.json \
    face_solution/configs/face_pose_lmk.json"

command_x3_common = \
    "echo 105000 >/sys/devices/virtual/thermal/thermal_zone0/trip_point_1_temp;\
    echo performance >/sys/devices/system/cpu/cpufreq/policy0/scaling_governor;\
    echo 0 > /sys/class/misc/ion/cma_carveout_size"
