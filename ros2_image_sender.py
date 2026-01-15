import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import time
import os
import numpy as np

class ImageSender(Node):
    def __init__(self):
        super().__init__('image_sender_node')
        # 修正1: 话题名必须与服务器一致 '/camera/image_raw'
        # 修正2: 类型必须是 Image，不能是 CompressedImage
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(0.5, self.timer_callback) # 0.5秒发一次
        
        self.img_path = '/home/unitree/poliformer/img.png'
        
        if not os.path.exists(self.img_path):
            self.get_logger().error(f"错误: 找不到图片 {self.img_path}")
            # 创建一个空的黑图防止报错
            self.cv_image = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            self.cv_image = cv2.imread(self.img_path)
            if self.cv_image is None:
                self.get_logger().error("错误: 无法读取图片")
                self.cv_image = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                self.get_logger().info(f"图片加载成功 (Numpy模式)")

    def timer_callback(self):
        try:
            # 重新读取图片（以防图片更新）
            if os.path.exists(self.img_path):
                img = cv2.imread(self.img_path)
                if img is not None:
                    self.cv_image = img

            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"
            
            height, width, channels = self.cv_image.shape
            msg.height = height
            msg.width = width
            msg.encoding = 'bgr8'
            msg.is_bigendian = 0
            msg.step = width * 3
            msg.data = self.cv_image.tobytes()
            
            self.publisher_.publish(msg)
            self.get_logger().info('正在发送图片 (Topic: /camera/image_raw)...')
        except Exception as e:
            self.get_logger().error(f'发送失败: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
