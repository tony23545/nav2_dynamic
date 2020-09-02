import numpy as np 
import cv2
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from nav2_dynamic_msgs.msg import ObjectCircle
from nav2_dynamic_msgs.srv import TrackCircle

import rclpy
from rclpy.node import Node


class Object:
    def __init__(self, pos, idx, r, dt = 0.33):
        
        self.pos = pos.reshape((2, 1))
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 2.0
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1.0
        self.kalman.statePost = np.concatenate([pos, np.zeros(2)]).astype(np.float32).reshape(4, 1)
        '''
        # 2nd order
        self.pos = pos.reshape((2, 1))
        self.kalman = cv2.KalmanFilter(6,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,dt,0,0.5*dt*dt,0],[0,1,0,dt,0,0.5*dt*dt],[0,0,1,0,dt,0],[0,0,0,1,0,dt],[0,0,0,0,1,0],[0,0,0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.eye(6, dtype = np.float32) * 3.0
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 2.0
        self.kalman.statePost = np.concatenate([pos, np.zeros(4)]).astype(np.float32).reshape(6, 1)
        '''
        self.dying = 0
        self.hit = False
        self.id = idx
        self.r = r
        self.vel = np.zeros(2)

    def predict(self):
        self.kalman.predict()
        self.pos = self.kalman.statePre[:2]

    def correct(self, measurement):
        self.kalman.correct(measurement)
        self.pos = self.kalman.statePost[:2]
        self.vel = self.kalman.statePost[2:4]

class Tracker(Node):
    def __init__(self):
        super().__init__('tracker_service')
        self.object_list = []
        self.max_id = 0
        self.srv = self.create_service(TrackCircle, 'track_circle_object_srv', self.update)

    def update(self, request, response):
        print("tracker update...")
        print(request.detections)
        detections = request.detections
        detect_list = []
        radius_list = []
        for det in detections:
            detect_list.append(np.array([det.x, det.y]))
            radius_list.append(det.r)
        num_of_object = len(self.object_list)
        num_of_detect = len(detect_list)

        for obj in self.object_list:
            obj.predict()

        cost = np.zeros((num_of_object, num_of_detect))

        for i in range(num_of_object):
            for j in range(num_of_detect):
                cost[i, j] = np.linalg.norm(self.object_list[i].pos.reshape(2) - detect_list[j])

        obj_ind, det_ind = linear_sum_assignment(cost)

        for o, d in zip(obj_ind, det_ind):
            self.object_list[o].correct(detect_list[d].astype(np.float32).reshape(2, 1))

        if num_of_object <= num_of_detect: # there are new detection
            self.birth(det_ind, num_of_detect, detect_list, radius_list)
            #TODO filter out high cost
        else:
            self.death(obj_ind, num_of_object)

        # construct response
        track_list = []
        for obj in self.object_list:
            ObjectMsg = ObjectCircle()
            ObjectMsg.x = np.float(obj.pos[0])
            ObjectMsg.y = np.float(obj.pos[1])
            ObjectMsg.vx = np.float(obj.vel[0])
            ObjectMsg.vy = np.float(obj.vel[1])
            ObjectMsg.r = obj.r
            ObjectMsg.id = obj.id
            track_list.append(ObjectMsg)

        response.tracking = track_list
        response.track_num = len(track_list)
        #print(response.tracking)
        return response


    def birth(self, det_ind, num_of_detect, detect_list, radius_list):
        for det in range(num_of_detect):
            if det not in det_ind:
                self.object_list.append(Object(detect_list[det], self.max_id, radius_list[det]))
                self.max_id += 1

    def death(self, obj_ind, num_of_object):
        new_object_list = []
        for obj in range(num_of_object):
            if obj not in obj_ind:
                self.object_list[obj].dying += 1
            else:
                self.object_list[obj].dying = 0

            if self.object_list[obj].dying < 2:
                new_object_list.append(self.object_list[obj])
        self.object_list = new_object_list

def main(args=None):
    rclpy.init(args=args)

    track_service = Tracker()
    print("running tracker service...")

    rclpy.spin(track_service)

    rclpy.shutdown()

if __name__ == "__main__":
    main()
