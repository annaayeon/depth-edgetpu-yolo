import os
import sys
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
import yaml
import timeit
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray,MultiArrayDimension
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from edgetpumodel import EdgeTPUModel
from utils import resize_and_pad, get_image_tensor, save_one_json, coco80_to_coco91_class

class priROS:
    def __init__(self):
        rospy.init_node('kudos_vision', anonymous = False)
        self.yolo_result_img_pub = rospy.Publisher("/output/image_raw2/compressed", CompressedImage, queue_size = 1)
    def yolo_result_img_talker(self, image_np,cam):
        import cv2
        print(np.shape(image_np)) 
        print("Mean FPS: {:1.2f}".format(fps))
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        self.yolo_result_img_pub.publish(msg)
#pub=rospy.Publisher('bounding_box_pub',Float32MultiArray,queue_size=1000)
#rate = rospy.Rate(10)
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser("EdgeTPU test runner")
    parser.add_argument("--model", "-m", help="weights file", required=True)
    parser.add_argument("--bench_speed", action='store_true', help="run speed test on dummy data")
    parser.add_argument("--bench_image", action='store_true', help="run detection test")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="model confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--names", type=str, default='data/coco.yaml', help="Names file")
    parser.add_argument("--image", "-i", type=str, help="Image file to run detection on")
    parser.add_argument("--device", type=int, default=0, help="Image capture device to run live detection")
    # Device num : v4l2-ctl --list-devices
    parser.add_argument("--stream", action='store_true', help="Process a stream")
    parser.add_argument("--bench_coco", action='store_true', help="Process a stream")
    parser.add_argument("--coco_path", type=str, help="Path to COCO 2017 Val folder")
    parser.add_argument("--quiet","-q", action='store_true', help="Disable logging (except errors)")
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='(optional) dataset.yaml path') 
        
    args = parser.parse_args()
    priROS = priROS()
    if args.quiet:
        logging.disable(logging.CRITICAL)
        logger.disabled = True
    
    if args.stream and args.image:
        logger.error("Please select either an input image or a stream")
        exit(1)
    
    model = EdgeTPUModel(args.model, args.names, conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh)
    input_size = model.get_image_size()

    x = (255*np.random.random((3,*input_size))).astype(np.uint8)
    model.forward(x)

    conf_thresh = 0.25
    iou_thresh = 0.45
    classes = None
    agnostic_nms = False
    max_det = 1000

    if args.bench_speed:
        logger.info("Performing test run")
        n_runs = 100
        
        
        inference_times = []
        nms_times = []
        total_times = []
        
        for i in tqdm(range(n_runs)):
            x = (255*np.random.random((3,*input_size))).astype(np.float32)
            
            pred = model.forward(x)
            tinference, tnms = model.get_last_inference_time()
            
            inference_times.append(tinference)
            nms_times.append(tnms)
            total_times.append(tinference + tnms)
            
        inference_times = np.array(inference_times)
        nms_times = np.array(nms_times)
        total_times = np.array(total_times)
            
        logger.info("Inference time (EdgeTPU): {:1.2f} +- {:1.2f} ms".format(inference_times.mean()/1e-3, inference_times.std()/1e-3))
        logger.info("NMS time (CPU): {:1.2f} +- {:1.2f} ms".format(nms_times.mean()/1e-3, nms_times.std()/1e-3))
        fps = 1.0/total_times.mean()
        logger.info("Mean FPS: {:1.2f}".format(fps))

    elif args.bench_image:
        logger.info("Testing on Zidane image")
        model.predict("./data/images/0483.jpg")

    elif args.bench_coco:
        logger.info("Testing on COCO dataset")
        
        model.conf_thresh = 0.001
        model.iou_thresh = 0.65
        
        coco_glob = os.path.join(args.coco_path, "*.jpg")
        images = glob.glob(coco_glob)
        
        logger.info("Looking for: {}".format(coco_glob))
        ids = [int(os.path.basename(i).split('.')[0]) for i in images]
        
        out_path = "./coco_eval"
        os.makedirs("./coco_eval", exist_ok=True)
        
        logger.info("Found {} images".format(len(images)))
        
        class_map = coco80_to_coco91_class()
        
        predictions = []
        
        for image in tqdm(images):
            res = model.predict(image, save_img=False, save_txt=False)
            save_one_json(res, predictions, Path(image), class_map)
            
        pred_json = os.path.join(out_path,
                    "{}_predictions.json".format(os.path.basename(args.model)))
        
        with open(pred_json, 'w') as f:
            json.dump(predictions, f,indent=1)
        
    elif args.image is not None:
        logger.info("Testing on user image: {}".format(args.image))
        model.predict(args.image)

    elif args.stream:
        logger.info("Opening stream on device: {}".format(args.device))
        total_times = []
        cam = cv2.VideoCapture(args.device)
        
        while True:
          try:
            res, image = cam.read()
            #msg=Float32MultiArray()
            #msg.layout.dim.append(MultiArrayDimension())
            #msg.layout.dim[0].label = "rows"
            #msg.layout.dim[0].size = 1000
            #msg.layout.dim[0].stride = 2
            #msg.layout.dim.append(MultiArrayDimension())
            #msg.layout.dim[1].label = "cols"
            #msg.layout.dim[1].size = 4
            #msg.layout.dim[1].stride = 3
            #msg.data=
            
            if res is False:
                logger.error("Empty image received")
                break
            else:
                total_times = []
                full_image, net_image, pad = get_image_tensor(image, input_size[0])
                pred = model.forward(net_image)
                tinference, tnms = model.get_last_inference_time()
                total_times=np.append(total_times,tinference + tnms)
                total_times = np.array(total_times)
                fps = 1.0/total_times.mean()
                # kudos_vision.py publish
                # Publish
                model.process_predictions(pred[0], full_image, pad)
                _,predimage=model.process_predictions(pred[0], full_image, pad)
                priROS.yolo_result_img_talker(predimage,fps)
                tinference, tnms = model.get_last_inference_time()
                logger.info("Frame done in {}".format(tinference+tnms))
          except KeyboardInterrupt:
            break
          
        cam.release()
            
        

        
    
