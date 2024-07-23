import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ur_custom_interfaces.msg import URCommand
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO
from message_filters import ApproximateTimeSynchronizer, Subscriber

class YOLOv8RealsenseNode(Node):
    def __init__(self):
        super().__init__('yolov8_realsense_node')

        self.color_sub = Subscriber(self, Image, 'camera/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, 'camera/camera/depth/image_rect_raw')

        self.ts = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.image_callback)

        self.publisher = self.create_publisher(URCommand, 'object_distance', 15)
        self.image_publisher = self.create_publisher(Image, 'camera/detections/image_raw', 10)
        self.bridge = CvBridge()
        self.yolo_model = YOLO('yolov8n.pt')  # Load YOLOv8 model
        self.get_logger().info('YOLOv8RealsenseNode has been started.')

    def image_callback(self, color_msg, depth_msg):
        self.get_logger().info('Received synchronized images')
        frame = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')

        self.get_logger().info('Converted ROS images to OpenCV images')
        results = self.yolo_model(frame)
        self.get_logger().info(f'YOLOv8 detection results: {results}')

        if results and results[0].boxes:
            for detection in results[0].boxes.data:
                self.get_logger().info(f'Detection: {detection}')

                # Extracting coordinates and class from the tensor
                x_min, y_min, x_max, y_max, conf, cls = detection[:6]
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                x_center = int((x_min + x_max) / 2)
                y_center = int((y_min + y_max) / 2)

                # Get class name from class index
                class_name = self.yolo_model.names[int(cls)]

                # Draw bounding box and label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                distance_msg = self.get_distance_to_object(depth_image, x_center, y_center)
                if distance_msg is not None:
                    # Publish distance only for books (class_ID = 73)
                    if int(cls) == 73:
                        self.publisher.publish(distance_msg)
                        self.get_logger().info(f'Published book distance: {distance_msg.z} meters')

                    # Add distance to the label
                    label = f'{class_name}: {distance_msg.z:.2f}m'
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                else:
                    self.get_logger().warn('Could not retrieve distance to object.')
                    cv2.putText(frame, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        else:
            self.get_logger().info('No detections found')

        # Publish the image with bounding boxes
        detection_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_publisher.publish(detection_image_msg)
        self.get_logger().info('Published detection image with bounding boxes')

    def get_distance_to_object(self, depth_image, x, y):
        if depth_image is None:
            self.get_logger().warn('Failed to convert depth image.')
            return None

        # Ensure x, y coordinates are within image bounds
        if y >= depth_image.shape[0] or x >= depth_image.shape[1]:
            self.get_logger().warn(f'Coordinates ({x}, {y}) out of bounds.')
            return None

        depth = depth_image[y, x]

        if np.isnan(depth) or np.isinf(depth):
            self.get_logger().warn(f'Invalid depth value at ({x}, {y})')
            return None

        msg = URCommand()
        msg.x = str(x) / 1000.0 # Cast x to float
        msg.y = str(y) / 1000.0  # Cast y to float
        msg.depth = str(depth) / 1000.0  # Convert from mm to meters if necessary

        return msg

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8RealsenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
