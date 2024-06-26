import time
import os
import logging
import rospy
import math
import yaml
import numpy as np
import pycoral.utils.edgetpu as etpu
from pycoral.adapters import common
from nms import non_max_suppression
import cv2
import json
import pyrealsense2 as rs

from utils import plot_one_box, Colors, get_image_tensor
from geometry_msgs.msg import Twist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EdgeTPUModel")
angle_pub = rospy.Publisher('move_tracking_Angle_pub', Twist, queue_size=1000)
goal_pub = rospy.Publisher('goal_position_pub', Twist, queue_size=1000)

class EdgeTPUModel:
    def __init__(self, model_file, names_file, cam_pipeline, conf_thresh=0.6, iou_thresh=0.45, filter_classes=None, agnostic_nms=False, max_det=1000):
        model_file = os.path.abspath(model_file)
        if not model_file.endswith('tflite'):
            model_file += ".tflite"
            
        self.model_file = model_file
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.filter_classes = filter_classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det

        self.no_ball_cnt = 0
        self.m_Pan_err = 0
        self.m_PanOffset = 0
        self.m_Tilt_err = 0
        self.m_TiltOffset = 0

        self.m_LeftLimit = 70
        self.m_RightLimit = -70
        self.m_TopLimit = 0
        self.m_BottomLimit = 90
            
        logger.info("Confidence threshold: {}".format(conf_thresh))
        logger.info("IOU threshold: {}".format(iou_thresh))
        
        self.inference_time = None
        self.nms_time = None
        self.interpreter = None
        self.colors = Colors()
        
        self.get_names(names_file)
        self.make_interpreter()
        self.get_image_size()        

        # RealSense 파이프라인 설정
        self.pipeline = cam_pipeline
        
    def __del__(self):
        self.pipeline.stop()

    def get_names(self, path):
        """
        Load a names file
        
        Inputs:
          - path: path to names file in yaml format
        """

        with open(path, 'r') as f:
          cfg = yaml.load(f, Loader=yaml.SafeLoader)
          
        names = cfg['names']
        logger.info("Loaded {} classes".format(len(names)))
        
        self.names = names
    
    def make_interpreter(self):
        """
        Internal function that loads the tflite file and creates
        the interpreter that deals with the EdgetPU hardware.
        """
        # Load the model and allocate
        self.interpreter = etpu.make_interpreter(self.model_file)
        self.interpreter.allocate_tensors()
    
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        logger.debug(self.input_details)
        logger.debug(self.output_details)
        
        self.input_zero = self.input_details[0]['quantization'][1]
        self.input_scale = self.input_details[0]['quantization'][0]
        self.output_zero = self.output_details[0]['quantization'][1]
        self.output_scale = self.output_details[0]['quantization'][0]
        
        # If the model isn't quantized then these should be zero
        # Check against small epsilon to avoid comparing float/int
        if self.input_scale < 1e-9:
            self.input_scale = 1.0
        
        if self.output_scale < 1e-9:
            self.output_scale = 1.0
    
        logger.debug("Input scale: {}".format(self.input_scale))
        logger.debug("Input zero: {}".format(self.input_zero))
        logger.debug("Output scale: {}".format(self.output_scale))
        logger.debug("Output zero: {}".format(self.output_zero))
        
        logger.info("Successfully loaded {}".format(self.model_file))
    
    def get_image_size(self):
        """
        Returns the expected size of the input image tensor
        """
        if self.interpreter is not None:
            self.input_size = common.input_size(self.interpreter)
            logger.debug("Expecting input shape: {}".format(self.input_size))
            return self.input_size
        else:
            logger.warn("Interpreter is not yet loaded")
            
    
    def predict(self, image_path, save_img=True, save_txt=True):
        logger.info("Attempting to load {}".format(image_path))
    
        full_image, net_image, pad = get_image_tensor(image_path, self.input_size[0])
        pred = self.forward(net_image)
        
        base, ext = os.path.splitext(image_path)
        
        output_path = base + "_detect" + ext
        det = self.process_predictions(pred[0], full_image, pad, output_path, save_img=save_img, save_txt=save_txt)
        
        return det
        
    
    def forward(self, x:np.ndarray, with_nms=True) -> np.ndarray:
        """
        Predict function using the EdgeTPU

        Inputs:
            x: (C, H, W) image tensor
            with_nms: apply NMS on output

        Returns:
            prediction array (with or without NMS applied)

        """
        tstart = time.time()
        # Transpose if C, H, W
        if x.shape[0] == 3:
          x = x.transpose((1,2,0))
        
        x = x.astype('float32')

        # Scale input, conversion is: real = (int_8 - zero)*scale
        x = (x/self.input_scale) + self.input_zero
        x = x[np.newaxis].astype(np.uint8)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        
        # Scale output
        result = (common.output_tensor(self.interpreter, 0).astype('float32') - self.output_zero) * self.output_scale
        self.inference_time = time.time() - tstart
        
        if with_nms:
        
            tstart = time.time()
            nms_result = non_max_suppression(result, self.conf_thresh, self.iou_thresh, self.filter_classes, self.agnostic_nms, max_det=self.max_det)
            self.nms_time = time.time() - tstart
            
            return nms_result
            
        else:    
          return result
          
    def get_last_inference_time(self, with_nms=True):
        """
        Returns a tuple containing most recent inference and NMS time
        """
        res = [self.inference_time]
        
        if with_nms:
          res.append(self.nms_time)
          
        return res
    
    def get_scaled_coords(self, xyxy, output_image, pad):
        """
        Converts raw prediction bounding box to orginal
        image coordinates.
        
        Args:
          xyxy: array of boxes
          output_image: np array
          pad: padding due to image resizing (pad_w, pad_h)
        """
        pad_w, pad_h = pad
        in_h, in_w = self.input_size
        out_h, out_w, _ = output_image.shape
                
        ratio_w = out_w/(in_w - pad_w)
        ratio_h = out_h/(in_h - pad_h) 
        
        out = []
        for coord in xyxy:

            x1, y1, x2, y2 = coord
                        
            x1 *= in_w*ratio_w
            x2 *= in_w*ratio_w
            y1 *= in_h*ratio_h
            y2 *= in_h*ratio_h
            
            x1 = max(0, x1)
            x2 = min(out_w, x2)
            
            y1 = max(0, y1)
            y2 = min(out_h, y2)
            
            out.append((x1, y1, x2, y2))
        return np.array(out).astype(int)


    def move_tracking(self, det):
        # 수직 시야각(VFOV) = 46도
        # 수평 시야각(HFOV) = 86.5도
        angle = [0,0]
        best_ball_det = max(det, key=lambda x: x[4])
        box_mx = (best_ball_det[0] + best_ball_det[2]) / 2
        box_my = (best_ball_det[1] + best_ball_det[3]) / 2
        ball_xy = [box_mx, box_my]

        err_X = box_mx - 320
        err_Y = box_my - 240
    
        m_Pan_p_gain = 0.05
        m_Pan_d_gain = 0.22
        m_Tilt_p_gain = 0.05
        m_Tilt_d_gain = 0.22

        m_Pan_err_diff = err_X - self.m_Pan_err
        self.m_Pan_err = err_X

        Pan_pOffset = self.m_Pan_err * m_Pan_p_gain
        Pan_pOffset *= Pan_pOffset
        if self.m_Pan_err < 0:
            Pan_pOffset = -Pan_pOffset

        Pan_dOffset = m_Pan_err_diff * m_Pan_d_gain
        Pan_dOffset *= Pan_dOffset
        if m_Pan_err_diff < 0:
            Pan_dOffset = -Pan_dOffset

        self.m_PanOffset = (Pan_pOffset + Pan_dOffset)
        m_PanAngle = self.m_PanOffset * (86.5 / 640)
        if m_PanAngle > self.m_LeftLimit:
            m_PanAngle = self.m_LeftLimit
        elif m_PanAngle < self.m_RightLimit:
            m_PanAngle = self.m_RightLimit

        m_Tilt_err_diff = err_Y - self.m_Tilt_err
        self.m_Tilt_err = err_Y

        Tilt_pOffset = self.m_Tilt_err * m_Tilt_p_gain
        Tilt_pOffset *= Tilt_pOffset
        if self.m_Tilt_err < 0:
            Tilt_pOffset = -Tilt_pOffset

        Tilt_dOffset = m_Tilt_err_diff * m_Tilt_d_gain
        Tilt_dOffset *= Tilt_dOffset
        if m_Tilt_err_diff < 0:
            Tilt_dOffset = -Tilt_dOffset

        self.m_TiltOffset = (Tilt_pOffset + Tilt_dOffset)
        m_TiltAngle = self.m_TiltOffset * (46 / 480)
        if m_TiltAngle > self.m_BottomLimit:
            m_TiltAngle = self.m_BottomLimit
        elif m_TiltAngle < self.m_TopLimit:
            m_TiltAngle = self.m_TopLimit

        angle[0], angle[1] = m_PanAngle, m_TiltAngle  
        return angle , ball_xy

    def goal_position_pub(self, det):
        best_goal_det = max(det, key=lambda x: x[4])
        best_goal_xy = [((best_goal_det[0] + best_goal_det[2]) / 2), ((best_goal_det[1] + best_goal_det[3]) / 2)]
        twist = Twist()
        twist.linear.x = best_goal_xy[0]
        twist.linear.y = best_goal_xy[1]
        goal_pub.publish(twist)

    def process_predictions(self, det, output_image, pad, output_path="detection.jpg", save_img=True, save_txt=True, hide_labels=False, hide_conf=False):
        xyxy = []
        angle = [0,0]
        twist = Twist()
        ball_flag = 0

        # RealSense 프레임 얻기
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

        # Numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        if len(det):
            det[:, :4] = self.get_scaled_coords(det[:, :4], output_image, pad)

            ball_det = [det[i, :] for i in range(len(det)) if det[i, 5] == 0 and det[i, 4] >= 0.8]
            goal_det = [det[i, :] for i in range(len(det)) if det[i, 5] == 1]
            foot_det = [det[i, :] for i in range(len(det)) if det[i, 5] == 2]

            output = {}
            base, ext = os.path.splitext(output_path)

            if len(ball_det):
                angle, ball_xy = self.move_tracking(ball_det)
                # 공 중심 좌표로 깊이 측정
                distance = depth_frame.get_distance(int(ball_xy[0]), int(ball_xy[1]))
                ball_distance = distance * 1000  # 미터를 밀리미터로 변환
                twist.linear.x = ball_distance
                ball_flag = 1
                if len(foot_det) and (ball_xy[1] > 300):
                    ball_flag = 2

            if len(goal_det):
                self.goal_position_pub(goal_det)

            s = ""
            for c in np.unique(det[:, -1]):
                n = (det[:, -1] == c).sum()
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
            if s != "":
                s = s.strip()
                s = s[:-1]
            logger.info(f"Detected: {s}")

            for *xyxy, conf, cls in reversed(det):
                if save_img:
                    c = int(cls)
                    label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                    output_image = plot_one_box(xyxy, output_image, label=label, color=self.colors(c, True))
                    best_ball_det = max(det, key=lambda x: x[4])

                    box_center_x = int((best_ball_det[0] + best_ball_det[2]) / 2)
                    box_center_y = int((best_ball_det[1] + best_ball_det[3]) / 2)
                    distance = depth_frame.get_distance(box_center_x, box_center_y)

                   # 깊이 정보를 화면에 표시
                    cv2.putText(output_image, f"{distance:.2f}m", (box_center_x, box_center_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    print('({},{}) distance = [{}m]'.format(box_center_x, box_center_y,distance))

                if save_txt:
                    xyxy.append(conf)
                    output[base] = {}
                    output[base]['box'] = xyxy
                    output[base]['conf'] = conf
                    output[base]['cls'] = cls
                    output[base]['cls_name'] = self.names[c]
            if save_txt:
                output_txt = base + ".txt"
                with open(output_txt, 'w') as f:
                    json.dump(output, f, indent=1)
            if save_img:
                cv2.imwrite(output_path, output_image)

        twist.angular.x = ball_flag
        twist.angular.y = angle[0]
        twist.angular.z = angle[1]
        angle_pub.publish(twist)
        cv2.imshow('Camera', output_image)
        cv2.waitKey(1)


        return det,output_image, xyxy