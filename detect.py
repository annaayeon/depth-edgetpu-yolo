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
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from edgetpumodel_change import EdgeTPUModel
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
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser("EdgeTPU test runner")
    parser.add_argument("--model", "-m", help="weights file", required=True)
    parser.add_argument("--bench_speed", action='store_true', help="run speed test on dummy data")
    parser.add_argument("--bench_image", action='store_true', help="run detection test")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="model confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--names", type=str, default='data/coco.yaml', help="Names file")
    parser.add_argument("--image", "-i", type=str, help="Image file to run detection on")
    parser.add_argument("--device", type=int, default=2, help="Image capture device to run live detection")
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

    # Set the desired frame size
    desired_width = 640
    desired_height = 480

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
        radii_history = []  # List to store the radii of detected circles over time        
        cam = cv2.VideoCapture(args.device)
        
        # Set the camera frame size
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
        
        start_time = time.time()  # Record the start time

        while time.time() - start_time < 20:  # Run the loop for 20 seconds
          try:
            res, image = cam.read()
            height, width = image.shape[:2]
            # new_img_size = (width, width)
            if res is False:
                logger.error("Empty image received")
                break
            else:
                
                # cameraMatrix 정의
                cameraMatrix = np.array([[9.4088597780774421e+02, 0., 3.7770158111216648e+02],
                                        [0., 9.4925081349933703e+02, 3.2918621409818121e+02],
                                        [0., 0., 1.]])

                # distCoeffs 정의
                distCoeffs = np.array([-4.4977607383629226e-01, -3.0529616557684319e-01,
                                    -3.9021603448837856e-03, -2.8130335366792153e-03,
                                    1.2224960045867554e+00])
                image = cv2.undistort(image, cameraMatrix, distCoeffs) #, None, new_img_size)

                total_times = []
                full_image, net_image, pad = get_image_tensor(image, input_size[0])
                pred = model.forward(net_image)
                tinference, tnms = model.get_last_inference_time()
                total_times=np.append(total_times,tinference + tnms)
                total_times = np.array(total_times)
                fps = 1.0/total_times.mean()
                # kudos_vision.py publish
                # Publish
                # model.process_predictions(pred[0], full_image, pad)

                _,predimage, bb=model.process_predictions(pred[0], full_image, pad)
                print("bounding box : ", bb)

                if len(bb) > 1:
                    x1, y1, x2, y2 = map(int, bb[:4])
                    x1 -= 10
                    y1 -= 10
                    x2 += 10
                    y2 += 10
                    roi = image[y1:y2, x1:x2]

                    # Convert ROI to grayscale
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                    # Create a binary mask for white ball
                    _, mask = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)

                    # Perform bitwise AND operation between the mask and ROI
                    masked_roi = cv2.bitwise_and(roi, roi, mask=mask)

                    # Detect circles using HoughCircles on the ROI
                    circles = cv2.HoughCircles(
                        gray_roi,
                        cv2.HOUGH_GRADIENT,
                        dp=1,  # Inverse ratio of the accumulator resolution to the image resolution
                        minDist=20,  # Minimum distance between the centers of the detected circles
                        param1=50,  # Upper threshold for the internal Canny edge detector
                        param2=30,  # Threshold for center detection
                        minRadius=5,  # Minimum radius of the detected circles
                        maxRadius=200  # Maximum radius of the detected circles
                    )

                    # Draw the largest two detected circles on the ROI
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        largest_circle = max(circles[0, :], key=lambda x: x[2])  # Select the largest circle by radius

                        center = (largest_circle[0], largest_circle[1])
                        radius = largest_circle[2]
                        radii_history.append(radius)  # Store the radius value in the history

                        # Draw the circle center
                        cv2.circle(roi, center, 1, (0, 100, 100), 3)
                        # Draw the circle outline
                        cv2.circle(roi, center, radius, (255, 0, 255), 3)

                        # Calculate circle area
                        area = np.pi * radius ** 2
                        print("Area of the ball:", area)

                    # Replace the processed ROI back into the full_image
                    full_image[y1:y2, x1:x2] = roi

                # Additional code to handle the end of the program, release the camera, and log the final inference time
                priROS.yolo_result_img_talker(predimage, fps)
                tinference, tnms = model.get_last_inference_time()
                logger.info("Frame done in {}".format(tinference+tnms))
                
          except KeyboardInterrupt:
            cam.release()
            break
          except:
            pass

        # Plot the radii of detected circles over time
        plt.plot(radii_history)
        plt.xlabel('Frame')
        plt.ylabel('Radius')
        plt.title('Changes in Detected Circle Radius over Time')
        plt.grid(True)
        plt.show()
        cam.release()
