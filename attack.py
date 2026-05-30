import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from protocol.msg import MotionServoCmd


def point_to_segment_dist(px, py, ax, ay, bx, by):
    """Point-to-segment distance used to detect whether the dog crosses the ball lane."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len2 = abx * abx + aby * aby
    if ab_len2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len2))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return math.hypot(px - proj_x, py - proj_y)

class Kick(Node):
    def __init__(self, name):
        super().__init__(name)
        self.dog_name = "Messi"
        self.dog_pose = None
        self.ball_pose = None
        self.goalA_pose = None
        self.dog_yaw = 0.0
        
        self.sub_dog = self.create_subscription(PoseStamped, '/vrpn/Messi_dog/pose', self.sub_callback_dog, 10)
        self.sub_ball = self.create_subscription(PoseStamped, '/vrpn/ball/pose', self.sub_callback_ball, 10) 
        self.sub_goalA = self.create_subscription(PoseStamped, '/vrpn/goal_left/pose', self.sub_callback_goalA, 10) 
        self.pub = self.create_publisher(MotionServoCmd, f'/{self.dog_name}/motion_servo_cmd', 10)
        self.timer = self.create_timer(0.02, self.timer_callback)
        
        self.log_counter = 0
        
        # 基础控制变量
        self.last_yaw_error = 0.0
        self.last_time = None
        self.last_turn = 0.0
        self.turn_lead_time = 0.06
        self.turn_static_offset = 0.04

        # 绕后平动专用参数（新增）
        self.back_kp_forward = 1.0   # 前后移动系数
        self.back_kp_side = 0.8      # 左右横移系数
        self.back_arrive_thresh = 0.15 # 到位判定距离

    def sub_callback_dog(self, msg):
        self.dog_pose = msg.pose.position
        q = msg.pose.orientation
        self.dog_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def sub_callback_ball(self, msg):
        self.ball_pose = msg.pose.position

    def sub_callback_goalA(self, msg):
        self.goalA_pose = msg.pose.position

    def timer_callback(self):
        if self.dog_pose is None or self.ball_pose is None or self.goalA_pose is None:
            return    
        
        # 时间间隔
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_time is None:
            dt = 0.02
            self.last_time = current_time
            return
        dt = max(0.01, min(current_time - self.last_time, 0.1))
        
        # =========================
        # 基础几何计算
        # =========================
        
        ball_to_goal_x = self.goalA_pose.x - self.ball_pose.x
        ball_to_goal_y = self.goalA_pose.y - self.ball_pose.y
        ball_dist_to_goal = math.sqrt(ball_to_goal_x**2 + ball_to_goal_y**2)
        
        if ball_dist_to_goal < 1e-6:
            return
            
        ball_to_goal_unit_x = ball_to_goal_x / ball_dist_to_goal
        ball_to_goal_unit_y = ball_to_goal_y / ball_dist_to_goal
        
        # =========================
        # 计算射门线垂直方向
        # =========================
        
        side_x = -ball_to_goal_unit_y
        side_y = ball_to_goal_unit_x
        
        # =========================
        # 动态决定用哪只脚
        # =========================
        
        if self.ball_pose.x > self.goalA_pose.x:
            push_side = -1.0
            foot_name = "右脚"
        else:
            push_side = 1.0
            foot_name = "左脚"
        
        # ============ 新增 ============
        kick_side = -push_side
        # ==============================
        
        # 调试打印
        self.log_counter += 1
        if self.log_counter % 25 == 0:
            self.get_logger().info(
                f"📍 球X={self.ball_pose.x:.2f} 球门X={self.goalA_pose.x:.2f} → {foot_name}"
            )
        
        # =========================
        # 狗与球的相对关系
        # =========================
        
        dog_to_ball_x = self.ball_pose.x - self.dog_pose.x
        dog_to_ball_y = self.ball_pose.y - self.dog_pose.y
        dist_to_ball = math.hypot(dog_to_ball_x, dog_to_ball_y)
        
        ball_to_dog_x = self.dog_pose.x - self.ball_pose.x
        ball_to_dog_y = self.dog_pose.y - self.ball_pose.y
        
        behind_metric = (
            ball_to_dog_x * ball_to_goal_unit_x +
            ball_to_dog_y * ball_to_goal_unit_y
        )
        
        # ============ 暴力推球模式 ============
        FORCE_PUSH_DIST = 1.5
        
        if ball_dist_to_goal < FORCE_PUSH_DIST:
            dx_ball = self.ball_pose.x - self.dog_pose.x
            dy_ball = self.ball_pose.y - self.dog_pose.y
            dist_to_ball = math.sqrt(dx_ball**2 + dy_ball**2)
            
            if ball_dist_to_goal > 0.01:
                if ball_dist_to_goal < 0.8:
                    push_offset = 0.05
                else:
                    push_offset = 0.07
                
                # ============ 改1：暴力推球模式 ============
                push_target_x = (
                    self.ball_pose.x 
                    + 0.3 * ball_to_goal_unit_x 
                    + kick_side * push_offset * side_x
                )
                push_target_y = (
                    self.ball_pose.y 
                    + 0.3 * ball_to_goal_unit_y 
                    + kick_side * push_offset * side_y
                )
                # ==========================================
            else:
                push_target_x = self.goalA_pose.x
                push_target_y = self.goalA_pose.y
            
            dx = push_target_x - self.dog_pose.x
            dy = push_target_y - self.dog_pose.y
            target_yaw = math.atan2(dy, dx)
            
            yaw_error = math.atan2(
                math.sin(target_yaw - self.dog_yaw),
                math.cos(target_yaw - self.dog_yaw)
            )
            
            Kp = 1.8
            Kd = 4.5
            yaw_error_dot = (yaw_error - self.last_yaw_error) / dt
            yaw_error_dot = max(-5.0,min(5.0, yaw_error_dot))

            turn = Kp * yaw_error + Kd * yaw_error_dot
            turn = max(min(turn, 3.0), -3.0)
            
            if abs(yaw_error) < 0.03:
                turn = 0.0
            
            if abs(yaw_error) < 0.3:
                speed = 1.2
            elif abs(yaw_error) < 0.6:
                speed = 0.8
            else:
                speed = 0.5
            
            id = 305
            cross = 0.0
            
            self.log_counter += 1
            if self.log_counter % 25 == 0:
                self.get_logger().info(
                    f"🔥暴力推球 [{foot_name}] 球离门={ball_dist_to_goal:.2f}m | "
                    f"E={yaw_error:.2f} Turn={turn:.2f} Spd={speed:.2f}"
                )
            
            self.last_yaw_error = yaw_error
            self.last_time = current_time
            
            msg = MotionServoCmd()
            msg.motion_id = int(f'{id}')
            msg.cmd_type = 1
            msg.value = 2
            msg.vel_des = [float(f'{speed:.2f}'), float(f'{cross:.2f}'), float(f'{turn:.2f}')]
            msg.step_height = [0.05, 0.05]
            self.pub.publish(msg)
            return
        
        # =========================
        # 核心修改：绕后/推进模式
        # =========================

        if behind_metric > 0.1:
            # =========================
            # 绕后模式（不变：仍然用 push_side）
            # =========================

            behind_dist = 0.9
            orbit_offset = 0.35

            # 绕后目标点（保持 push_side 不变）
            target_x = (
                self.ball_pose.x
                - behind_dist * ball_to_goal_unit_x
                + push_side * orbit_offset * side_x
            )

            target_y = (
                self.ball_pose.y
                - behind_dist * ball_to_goal_unit_y
                + push_side * orbit_offset * side_y
            )

            # =========================
            # 狗到球距离
            # =========================

            dog_ball_dx = self.ball_pose.x - self.dog_pose.x
            dog_ball_dy = self.ball_pose.y - self.dog_pose.y

            dog_ball_dist = math.hypot(
                dog_ball_dx,
                dog_ball_dy
            )

            # =========================
            # 绕球切向速度场
            # 越靠近球，绕行越强
            # =========================

            orbit_gain = 0.0

            if dog_ball_dist < 0.9:
                orbit_gain = (0.9 - dog_ball_dist) * 1.8

            # 切向方向（沿射门线垂直方向绕，保持 push_side 不变）
            orbit_vx = push_side * side_x * orbit_gain
            orbit_vy = push_side * side_y * orbit_gain

            # =========================
            # 修改后的目标误差
            # =========================

            err_x = (
                target_x - self.dog_pose.x
                + orbit_vx
            )

            err_y = (
                target_y - self.dog_pose.y
                + orbit_vy
            )

            # =========================
            # 身体朝向：绕后时始终看着球
            # =========================

            target_yaw = math.atan2(
                self.ball_pose.y - self.dog_pose.y,
                self.ball_pose.x - self.dog_pose.x
            )

            yaw_error = math.atan2(
                math.sin(target_yaw - self.dog_yaw),
                math.cos(target_yaw - self.dog_yaw)
            )

            yaw_error_dot = (
                yaw_error - self.last_yaw_error
            ) / dt

            yaw_error_dot = max(
                -4.0,
                min(4.0, yaw_error_dot)
            )

            Kp_turn = 1.0
            Kd_turn = 0.8

            turn = (
                Kp_turn * yaw_error
                + Kd_turn * yaw_error_dot
            )

            turn = max(min(turn, 1.2), -1.2)

            # =========================
            # 机器狗自身坐标系
            # =========================

            forward_x = math.cos(self.dog_yaw)
            forward_y = math.sin(self.dog_yaw)

            side_x_body = -math.sin(self.dog_yaw)
            side_y_body = math.cos(self.dog_yaw)

            err_forward = (
                err_x * forward_x
                + err_y * forward_y
            )

            err_side = (
                err_x * side_x_body
                + err_y * side_y_body
            )

            # =========================
            # 平动控制
            # =========================

            speed = self.back_kp_forward * err_forward
            cross = self.back_kp_side * err_side

            speed = max(min(speed, 0.7), -0.7)
            cross = max(min(cross, 0.75), -0.75)

            if dog_ball_dist < 0.45:
                speed *= 0.7

            if self.log_counter % 25 == 0:
                self.get_logger().info(
                    f"↩️绕球模式 | "
                    f"D={dog_ball_dist:.2f} "
                    f"Orbit={orbit_gain:.2f} "
                    f"F={speed:.2f} "
                    f"C={cross:.2f} "
                    f"T={turn:.2f}"
                )

            self.last_turn = turn
            self.last_yaw_error = yaw_error
            self.last_time = current_time

            msg = MotionServoCmd()
            msg.motion_id = 305
            msg.cmd_type = 1
            msg.value = 2

            msg.vel_des = [
                float(f'{speed:.2f}'),
                float(f'{cross:.2f}'),
                float(f'{turn:.2f}')
            ]

            msg.step_height = [0.05, 0.05]

            self.pub.publish(msg)

            return

        else:
            # ---------------------
            # 正常带球推进
            # ---------------------
            push_offset = 0.05
            # ============ 改2：正常推进模式 ============
            target_x = (
                self.ball_pose.x
                + 0.28 * ball_to_goal_unit_x
                + kick_side * push_offset * side_x
            )
            target_y = (
                self.ball_pose.y
                + 0.28 * ball_to_goal_unit_y
                + kick_side * push_offset * side_y
            )
            # ==========================================
            max_speed = 1.25

        # =========================
        # 朝向控制
        # =========================

        dx = target_x - self.dog_pose.x
        dy = target_y - self.dog_pose.y

        target_yaw = math.atan2(dy, dx)

        yaw_error = math.atan2(
            math.sin(target_yaw - self.dog_yaw),
            math.cos(target_yaw - self.dog_yaw)
        )

        # =========================
        # PD 转向
        # =========================

        yaw_error_dot = (yaw_error - self.last_yaw_error) / dt
        yaw_error_dot = max(-5.0,min(5.0,yaw_error_dot))

        Kp = 1.6
        Kd = 1.5

        turn = (
            Kp * yaw_error
            + Kd * yaw_error_dot
        )

        turn = max(min(turn, 2.2), -2.2)

        turn = (
            0.6 * self.last_turn
            + 0.4 * turn
        )

        if abs(turn) < 0.03:
            turn = 0.0

        self.last_turn = turn
        self.last_yaw_error = yaw_error
        self.last_time = current_time

        # =========================
        # 速度控制
        # =========================

        speed_scale = max(0.35,math.exp(-0.8 * abs(yaw_error)))
        speed = max_speed * speed_scale

        speed *= max(0.75,1.0 - 0.08 * abs(turn))

        if dist_to_ball < 0.2:
            speed *= 0.7

        if behind_metric < 0 and speed < 0.15:
            speed = 0.15

        cross = 0.0
        id = 305

        # =========================
        # 发送指令
        # =========================
        
        msg = MotionServoCmd()
        msg.motion_id = int(f'{id}')
        msg.cmd_type = 1
        msg.value = 2
        msg.vel_des = [float(f'{speed:.2f}'), float(f'{cross:.2f}'), float(f'{turn:.2f}')]
        msg.step_height = [0.05, 0.05]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Kick("kick")

    stand_msg = MotionServoCmd()
    stand_msg.motion_id = 101
    stand_msg.cmd_type = 1
    stand_msg.value = 2
    stand_msg.vel_des = [0.0, 0.0, 0.0]
    stand_msg.step_height = [0.05, 0.05]
    for _ in range(5):
        node.pub.publish(stand_msg)
        rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().info("✅ 已发送起立指令")

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
