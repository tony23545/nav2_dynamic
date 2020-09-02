import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import from common libraries
import numpy as np 
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2 
import numpy as np
import matplotlib.pyplot as plt
from nav2_dynamic_msgs.msg import ObjectCircle
from nav2_dynamic_msgs.srv import TrackCircle

from threading import Thread

class detectron_srv(Node):
    def __init__(self):
        super().__init__('detectron_node')
        # setup detectron model
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)
        #self.srv = self.create_service(DetecSrv, 'detection_srv', self.runPredictor)

        # subscribe to sensor 
        self.subscription = self.create_subscription(
            PointCloud2,
            '/realsense/camera/pointcloud',
            self.callback,
            30)
        self.count = -1

        # wait srv
        self.cli = self.create_client(TrackCircle, 'track_circle_object_srv')       # CHANGE
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.color_list = ['r', 'g', 'b', 'y', 'k', 'c']
        self.new_frame = False

    def outlier_filter(self, x, z, idx):
        x_mean = np.mean(x)
        x_var = np.var(x)
        z_mean = np.mean(z)
        z_var = np.var(z)
        gaussian_kernel = np.exp(-0.5 * (np.power(x-x_mean, 2) / x_var + np.power(z-z_mean, 2) / z_var)) / (2 * np.pi * np.sqrt(x_var * z_var))
        return idx[gaussian_kernel > 0.5]


    def callback(self, msg):
        if self.new_frame:
            return

        print("processing one frame...")
        self.new_frame = True

        height = msg.height
        width = msg.width
        points = np.array(msg.data, dtype = 'uint8')
        # rgb image
        rgb_offset = msg.fields[3].offset
        point_step = msg.point_step
        r = points[rgb_offset::point_step]
        g = points[(rgb_offset+1)::point_step]
        b = points[(rgb_offset+2)::point_step]
        self.img = np.concatenate([r[:, None], g[:, None], b[:, None]], axis = -1)
        self.img = self.img.reshape((height, width, 3))
        # point cloud
        points = points.view('<f4')
        down_sample_scale = 16
        x = points[::int(down_sample_scale  * point_step / 4)]
        y = points[1::int(down_sample_scale * point_step / 4)]
        z = points[2::int(down_sample_scale * point_step / 4)]

        # call detectron model
        outputs = self.predictor(self.img)

        # map to point cloud data
        color = np.zeros_like(x, dtype = 'uint8')
        num_classes = outputs['instances'].pred_classes.shape[0]
        masks = outputs["instances"].pred_masks.cpu().numpy().astype('uint8').reshape((num_classes, -1))[:, ::down_sample_scale]
        head_count = 0

        TrackReq = TrackCircle.Request()
        detections = []
        for i in range(num_classes):
            if outputs["instances"].pred_classes[i] == 0:
                idx = np.where(masks[i])[0]
                idx = self.outlier_filter(x[idx], z[idx], idx)
                if idx.shape[0] == 0:
                    continue
                ObjectMsg = ObjectCircle()
                ObjectMsg.x = np.float(x[idx].mean())
                ObjectMsg.y = np.float(z[idx].mean())
                ObjectMsg.r = np.linalg.norm(np.concatenate([x[idx, None], z[idx, None]], axis = -1) - np.array([[ObjectMsg.x, ObjectMsg.y]]), axis = -1).max()
                detections.append(ObjectMsg)
                head_count += 1
                #color[idx] += head_count

        TrackReq.detections = detections
        TrackReq.detect_num = head_count

        print("calling tracker service... ")
        self.future = self.cli.call_async(TrackReq)
        

    def visualization(self):
        TrackRes = self.future.result()
        plt.clf()
        fig=plt.figure(1)
        ax=fig.add_subplot(1,2,1)
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 5)
        for i in range(TrackRes.track_num):
            obj = TrackRes.tracking[i]
            circ = plt.Circle((obj.x, obj.y), obj.r, color = self.color_list[obj.id], fill=False)
            ax.add_patch(circ)
            ax.arrow(obj.x, obj.y, obj.vx, obj.vy, color = self.color_list[obj.id])
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(np.flip(self.img, 2))
        plt.draw()
        plt.pause(0.01)
        '''
        plt.subplot(1, 2, 1)
        #plt.scatter(x, z, c = color, s = 0.1)
        for i in range(TrackRes.track_num):
            obj = TrackRes.tracking[i]
            plt.Circle((obj.x, obj.y), obj.r, color = self.color_list[obj.id])
        plt.subplot(1, 2, 2)
        plt.imshow(np.flip(self.img, 2))
        plt.draw()
        plt.pause(0.01)'''


    def spin(self):
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.new_frame and self.future.done():
                self.visualization()
                self.new_frame = False

def main():
    rclpy.init(args = None)
    subs = detectron_srv()
    print("start spining detectron_srv node...")
    
    subs.spin()
    subs.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()