#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class InteractiveMover(Node):
    def __init__(self):
        super().__init__('interactive_mover')
        # ★ 直接发布到 /joint_states，这是 robot_state_publisher 和 Foxglove 默认订阅的话题
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.1, self.publish_state)
        self.joint_names = [
            "base_pitch_joint",
            "boom_pitch_joint",
            "swing_joint",
            "feed_rot_joint",
            "feed_prismatic_joint"
        ]
        self.positions = [0.0, 0.0, 0.0, 0.0, 0.0]

    def publish_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.positions
        self.pub.publish(msg)

def main():
    rclpy.init()
    mover = InteractiveMover()
    print("\n机械臂直接控制已启动！")
    print("关节顺序: 基座俯仰, 大臂俯仰, 左右旋转, 推进梁旋转, 推进梁伸缩")
    print("输入5个值（度），空格分隔，例如: 40 30 0 0 2.0")
    print("输入 q 退出。\n")

    while rclpy.ok():
        try:
            inp = input(">> ").strip()
            if inp.lower() == 'q':
                break
            vals = [float(x) for x in inp.split()]
            if len(vals) == 5:
                mover.positions = [v * 0.0174533 for v in vals]
                print(f"已设置: 基座 {vals[0]}° | 大臂 {vals[1]}° | 左右 {vals[2]}° | 旋转 {vals[3]}° | 伸缩 {vals[4]}m")
            else:
                print("请精确输入5个数值！")
        except KeyboardInterrupt:
            break
        except ValueError:
            print("格式错误，请重新输入。")

    mover.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
