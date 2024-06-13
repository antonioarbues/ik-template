import rclpy
import pinocchio as pin
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory
from numpy.linalg import norm, solve
from copy import deepcopy

# Robots setup
LEADER_PACKAGE_NAME = 'leader'
FOLLOWER_PACKAGE_NAME = 'follower'
FOLLOWER_EE_LINK_NAME = 'follower_ee_link'
JOINT_IDS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
EE_JOINT_ID = 6

# IK parameters
eps = 1e-2  # convergence threshold
IT_MAX = 100  # max number of iterations
DT = 5e-2  # integration step
damp = 1e-12  # damping factor
filter_gain_v = 0.1  # joint velocity filter gain

class IK(Node):
    def __init__(self):
        super().__init__('ik_node')
        urdf_filename = get_package_share_directory(FOLLOWER_PACKAGE_NAME) + '/urdf/follower.urdf'
        self.model = pin.buildModelFromUrdf(urdf_filename)
        self.data = self.model.createData()
        self._ee_frame = self.model.getFrameId(FOLLOWER_EE_LINK_NAME)
        self.q = None
        self.oMdes = None
        
        # Joint velocity filtering
        self.prev_v = None

        self.subscription_target = self.create_subscription(
            PoseStamped,
            '/leader/ee_pose',
            self.leader_pose_callback,
            1)

        self.subscription_follower_state = self.create_subscription(
            JointState,
            '/follower/joint_states',
            self.joint_states_callback,
            1)

        # Create timer for joint state publisher
        self.timer = self.create_timer(0.01, self.ik)

        # Joint state publisher
        self.joint_targets_publisher = self.create_publisher(JointState, '/follower/joint_targets', 10)
    
    def joint_states_callback(self, msg: JointState):
        self.q = np.array(msg.position)

    def leader_pose_callback(self, msg: PoseStamped):
        position = msg.pose.position
        orientation = msg.pose.orientation
        quaternion = pin.Quaternion(orientation.w, orientation.x, orientation.y, orientation.z)

        t_target = np.array([position.x, position.y, position.z])
        M_target = quaternion.matrix()
        self.oMdes = pin.SE3(M_target, t_target)  # desired pose in world frame

    def ik(self,):
        """
        data.oMi = Vector of absolute joint placements (wrt the world), SE3
        data.oMf = Vector of absolute operationnel frame placements (wrt the world), SE3
        data.iMf = Vector of joint placements wrt to algorithm end effector, SE3

        log6 = transforms a rigid transformation represented by an homogeneous matrix (M) into a 6D vector (v, w)
        Jlog6 = computes the derivative of log6

        aXb.act(by) -> ay (rigid transformation)
        aXb.actInv(ay) -> by (rigid transformation inverting the transformation matrix)
        """
        if self.q is None or self.oMdes is None:
            return

        q = deepcopy(self.q)
        for i in range(IT_MAX):
            pin.forwardKinematics(self.model, self.data, q)
            iMd = self.data.oMi[EE_JOINT_ID].actInv(self.oMdes)
            err = pin.log(iMd).vector  # in joint frame
            J = pin.computeJointJacobian(self.model, self.data, q, EE_JOINT_ID)  # in joint frame
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(len(JOINT_IDS)), err))
            self.prev_v = v if self.prev_v is None else filter_gain_v * v + (1 - filter_gain_v) * self.prev_v  # filter joint velocity
            v = self.prev_v
            q = pin.integrate(self.model, q, v * DT)

            if norm(err) < eps:
                break

        joint_targets_msg = JointState()
        joint_targets_msg.header.stamp = self.get_clock().now().to_msg()
        joint_targets_msg.name = JOINT_IDS
        joint_targets_msg.position = q.flatten().tolist()
        self.joint_targets_publisher.publish(joint_targets_msg)


def main(args=None):
    rclpy.init(args=args)
    node = IK()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
