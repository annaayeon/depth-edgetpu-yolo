import argparse
import logging
import numpy as np
import cv2
import rospy
import sys  # sys 모듈을 가져옵니다.
from std_msgs.msg import Float64
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from tensorflowmodel import TensorFlowModel
from utils import get_image_tensor, resize_and_pad, coco80_to_coco91_class, save_one_json, StreamingDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class priROS:
    def __init__(self, model_path, conf_thresh, iou_thresh, names):
        self.model = TensorFlowModel(model_path, names, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        self.bridge = CvBridge()
        self.detection_result_img_pub = rospy.Publisher("/output/image_raw2/compressed", CompressedImage, queue_size=1)
        self.distance_pub = rospy.Publisher("/distance_topic", Float64, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.image_callback)

    def detection_result_img_talker(self, image_np, fps):
        print("Mean FPS: {:1.2f}".format(fps))
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        self.detection_result_img_pub.publish(msg)

    def distance_talker(self, distance):
        self.distance_pub.publish(Float64(distance))

    def image_callback(self, data):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
            cv_image = cv2.resize(cv_image,(640,480))
            full_image, net_image, pad = get_image_tensor(cv_image, self.model.get_image_size()[0])
            pred = self.model.forward(net_image)
            # process_predictions 함수의 반환값을 확인하고 처리합니다.
            result = self.model.process_predictions(pred[0], full_image, pad)
            if result is None:
                logger.error("No predictions to process.")
                return  # 함수가 None을 반환하면 여기서 처리하고 빠져나갑니다.
            _, predimage, _ = result
            fps = 1.0 / self.model.get_last_inference_time()[0]
            self.detection_result_img_talker(predimage, fps)
            tinference, tnms = self.model.get_last_inference_time()
            logger.info("Frame processed in {}".format(tinference + tnms))
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    rospy.init_node('vision', anonymous=False)
    parser = argparse.ArgumentParser("TensorFlow model test runner")
    parser.add_argument("--model", "-m", help="weights file", required=True)
    parser.add_argument("--conf_thresh", type=float, default=0.6, help="model confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--names", type=str, default='data/coco.yaml', help="Names file")
    parser.add_argument("--quiet", "-q", action='store_true', help="Disable logging (except errors)")

    # argparse의 parse_known_args 메서드에 수정된 sys.argv를 전달합니다.
    args, unknown = parser.parse_known_args(rospy.myargv(argv=sys.argv)[1:])

    if args.quiet:
        logging.disable(logging.CRITICAL)
        logger.disabled = True

    vision_node = priROS(args.model, args.conf_thresh, args.iou_thresh, args.names)
    rospy.spin()
