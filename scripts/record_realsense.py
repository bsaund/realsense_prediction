#! /usr/bin/env python3
import yaml
import rospkg
import rospy
from pathlib import Path
from sensor_msgs.msg import CompressedImage, Imu
import cv2
from realsense_prediction import datatools

SAVE_IMAGES = True

has_saved_img = False
save_path = Path('/tmp')
img_num = 1


def add_dataseq_line(txt):
    with open(save_path / 'dataseq.txt', 'a') as f:
        f.write(f'{txt}\n')


def save_image_callback(img_msg: CompressedImage):
    global img_num
    print("Got image message")
    # cv2.imdecode(img_msg.data)
    # if has_saved_img:
    #     return
    if SAVE_IMAGES:
        with open(save_path / f'{img_num:05}.jpg', 'wb') as f:
            f.write(img_msg.data)
    add_dataseq_line(f'{img_msg.header.stamp}; image; {img_num:05}.jpg')
    img_num += 1


def save_accel_data(accel_msg: Imu):
    la = accel_msg.linear_acceleration
    accel_str = f'{la.x}, {la.y}, {la.z}'
    add_dataseq_line(f'{accel_msg.header.stamp}; accel; {accel_str}')


def save_gyro_data(accel_msg: Imu):
    av = accel_msg.angular_velocity
    av_str = f'{av.x}, {av.y}, {av.z}'
    add_dataseq_line(f'{accel_msg.header.stamp}; gyro; {av_str}')


def get_save_directory():
    global save_path
    # config_fp = Path(rospkg.RosPack().get_path('realsense_prediction')) / 'config.yaml'
    # with config_fp.open() as f:
    #     config = yaml.load(f, yaml.FullLoader)
    config = datatools.load_config()
    save_path = Path(config['save_path'])
    max_video = max([int(p.parts[-1].strip('video_')) for p in save_path.glob('video*/**')] + [0])
    save_path = save_path / f'video_{max_video + 1:03}'
    save_path.mkdir()


def main():
    rospy.init_node('record_realsense')
    get_save_directory()

    rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, save_image_callback)
    rospy.Subscriber('/camera/accel/sample', Imu, save_accel_data)
    rospy.Subscriber('/camera/gyro/sample', Imu, save_gyro_data)

    rospy.spin()


if __name__ == "__main__":
    main()

print("hello")
