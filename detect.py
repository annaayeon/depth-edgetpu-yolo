import os
import argparse
import logging
import time
from pathlib import Path
import glob
import json
import rospy
import numpy as np
from tqdm import tqdm
import cv2
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
import pyrealsense2 as rs 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from edgetpumodel import EdgeTPUModel
from utils import resize_and_pad, get_image_tensor, save_one_json, coco80_to_coco91_class, StreamingDataProcessor

class priROS:
    def __init__(self):
        rospy.init_node('kudos_vision', anonymous=False)
        self.yolo_result_img_pub = rospy.Publisher("/output/image_raw2/compressed", CompressedImage, queue_size=1)
        self.distance_pub = rospy.Publisher("/distance_topic", Float64, queue_size=1)  # Adjust topic name and message type

    def yolo_result_img_talker(self, image_np, fps):
        import cv2
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        self.yolo_result_img_pub.publish(msg)

    def distance_talker(self, distance):
        self.distance_pub.publish(Float64(distance))  # Adjust message type if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser("EdgeTPU test runner")
    parser.add_argument("--model", "-m", help="weights file", required=True)
    parser.add_argument("--conf_thresh", type=float, default=0.6, help="model confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--names", type=str, default='data/coco.yaml', help="Names file")
    parser.add_argument("--quiet", "-q", action='store_true', help="Disable logging (except errors)")
    parser.add_argument("--stream", action='store_true', help="Process a stream")
    args = parser.parse_args()

    priROS = priROS()
    if args.quiet:
        logging.disable(logging.CRITICAL)
        logger.disabled = True

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    model = EdgeTPUModel(args.model, args.names, conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh, cam_pipeline=pipeline)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            total_times = []
            full_image, net_image, pad = get_image_tensor(color_image, model.get_image_size()[0])
            pred = model.forward(net_image)
            tinference, tnms = model.get_last_inference_time()
            total_times = np.append(total_times, tinference + tnms)
            total_times = np.array(total_times)
            fps = 1.0 / total_times.mean()

            _, predimage, bb = model.process_predictions(pred[0], full_image, pad)

            priROS.yolo_result_img_talker(predimage, fps)
            tinference, tnms = model.get_last_inference_time()
            logger.info("Frame done in {}".format(tinference + tnms))

            if cv2.waitKey(1) == 27:
                break


    finally:
        model.pipeline.stop()
