import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageReceiver(Node):
    def __init__(self):
        super().__init__('image_receiver_node')
        
        # 1. 这里的名字必须和 ros2 topic echo 看到的一模一样
        self.topic_name = '/camera/image_raw'
        
        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.listener_callback,
            10)
        
        self.bridge = CvBridge()
        
        # 2. 设置保存路径 (对应宿主机的 /home/jitl/PoliFormer/data_from_dog)
        self.save_dir = '/root/code/data_from_dog'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"创建文件夹: {self.save_dir}")
            
        print(f"正在监听话题: {self.topic_name} ...")
        print(f"图片将保存到: {self.save_dir}")

    def listener_callback(self, msg):
        try:
            # 3. 将 ROS 消息转为 OpenCV 图片
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 4. 生成文件名并保存
            timestamp = msg.header.stamp.sec
            filename = os.path.join(self.save_dir, f"received_{timestamp}.png")
            
            cv2.imwrite(filename, cv_image)
            self.get_logger().info(f'成功保存图片: {filename}')
            
        except Exception as e:
            self.get_logger().error(f'处理失败: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
