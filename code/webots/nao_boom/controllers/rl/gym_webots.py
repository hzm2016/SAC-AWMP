import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
from controller import Robot, Supervisor


class Nao(gym.Env):
    """
        Y axis is the vertical axis.
        Base class for Webots actors in a Scene.
        These environments create single-player scenes and behave like normal Gym environments, if
        you don't use multiplayer.
    """

    electricity_cost = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    joints_at_limit_cost = -0.2  # discourage stuck joints


    frame = 0
    _max_episode_steps = 1000

    initial_y = None
    body_xyz = None
    joint_angles = None


    def __init__(self, action_dim, obs_dim):
        self.robot = Supervisor()

        self.robot_node = self.robot.getFromDef('Nao')
        self.robot_trans_field = self.robot_node.getField("translation")
        self.robot_rot_field = self.robot_node.getField("rotation")
        self.robot_ini_trans = self.robot_trans_field.getSFVec3f()
        self.robot_ini_rot = self.robot_rot_field.getSFRotation()

        self.boom_body = self.robot.getFromDef('BoomBody')
        self.boom_body_trans_field = self.boom_body.getField("translation")
        self.boom_body_rot_field = self.boom_body.getField("rotation")
        self.boom_body_ini_trans = self.boom_body_trans_field.getSFVec3f()
        self.boom_body_ini_rot = self.boom_body_rot_field.getSFRotation()


        self.boom_base = self.robot.getFromDef('BoomBase')
        self.boom_base_trans_field = self.boom_base.getField("translation")
        self.timeStep = int(self.robot.getBasicTimeStep()) # ms
        self.find_and_enable_devices()

        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)


    def find_and_enable_devices(self):

        # inertial unit
        self.inertial_unit = self.robot.getInertialUnit("inertial unit")
        self.inertial_unit.enable(self.timeStep)

        # gps
        self.gps = self.robot.getGPS("gps")
        self.gps.enable(self.timeStep)

        # foot sensors

        self.fsr = [self.robot.getTouchSensor("RFsr"), self.robot.getTouchSensor("LFsr")]
        for i in range(len(self.fsr)):
            self.fsr[i].enable(self.timeStep)

        # all motors
        motor_names = [# 'HeadPitch', 'HeadYaw',
                       'LAnklePitch', 'LAnkleRoll', 'LKneePitch',
                       'LHipPitch', 'LHipRoll', 'LHipYawPitch',
                       'RAnklePitch', 'RAnkleRoll', 'RKneePitch',
                       'RHipPitch', 'RHipRoll', 'RHipYawPitch',
                       ]
        self.motors = []
        for i in range(len(motor_names)):
            self.motors.append(self.robot.getMotor(motor_names[i]))

        # leg pitch motors
        self.legPitchMotor = [self.robot.getMotor('RHipPitch'),
                              self.robot.getMotor('RKneePitch'),
                              self.robot.getMotor('RAnklePitch'),
                              self.robot.getMotor('LHipPitch'),
                              self.robot.getMotor('LKneePitch'),
                              self.robot.getMotor('LAnklePitch')]

        # leg pitch sensors
        self.legPitchSensor =[self.robot.getPositionSensor('RHipPitchS'),
                              self.robot.getPositionSensor('RKneePitchS'),
                              self.robot.getPositionSensor('RAnklePitchS'),
                              self.robot.getPositionSensor('LHipPitchS'),
                              self.robot.getPositionSensor('LKneePitchS'),
                              self.robot.getPositionSensor('LAnklePitchS')]
        for i in range(len(self.legPitchSensor)):
            self.legPitchSensor[i].enable(self.timeStep)


    def apply_action(self, a):
        assert (np.isfinite(a).all())
        for n, j in enumerate(self.legPitchMotor):
            joint_angle = self.read_joint_angle(joint_idx=n)
            max_joint_angle = j.getMaxPosition()
            min_joint_angle = j.getMinPosition()
            if joint_angle > max_joint_angle:
                j.setPosition(max_joint_angle - 0.1)
            elif joint_angle < min_joint_angle:
                j.setPosition(min_joint_angle + 0.1)
            else:
                j.setTorque(1.0 * j.getMaxTorque() * float(np.clip(a[n], -1, +1)))


    def read_joint_angle(self, joint_idx):
        joint_angle = self.legPitchSensor[joint_idx].getValue() % (2.0 * np.pi)
        if joint_angle > np.pi:
            joint_angle -= 2.0 * np.pi
        return joint_angle

    def calc_state(self):
        joint_states = np.zeros(2*len(self.legPitchMotor))
        # even elements [0::2] position, scaled to -1..+1 between limits
        for r in range(6):
            joint_angle = self.read_joint_angle(joint_idx=r)
            #
            # max_joint_angle = self.legPitchMotor[r].getMaxPosition()
            # min_joint_angle = self.legPitchMotor[r].getMinPosition()
            # # print('joint_angle: {}, max_q_{}, min_q_{}'.format(joint_angle, max_joint_angle, min_joint_angle))
            # joint_states[2 * r] = -(joint_angle - 0.5 * (max_joint_angle + min_joint_angle)) \
            #                   / (0.5 * (max_joint_angle - min_joint_angle))
            # if r in [1, 4]: # only the direction of the knee is the same as human
            #     joint_states[2 * r] = -joint_states[2 * r]
            if r in [0, 3]:
                joint_states[2 * r] = (-joint_angle - np.deg2rad(35)) / np.deg2rad(80)
            elif r in [1, 4]:
                joint_states[2 * r] = 1 - joint_angle / np.deg2rad(75)
            elif r in [2, 5]:
                joint_states[2 * r] = -joint_angle / np.deg2rad(45)
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        for r in range(6):
            if self.joint_angles is None:
                joint_states[2 * r + 1] = 0.0
            else:
                joint_states[2 * r + 1] = 0.5 * (joint_states[2*r] - self.joint_angles[r])

        self.joint_angles = np.copy(joint_states[0::2])
        self.joint_speeds = joint_states[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(joint_states[0::2]) > 0.99)

        if self.body_xyz is None:
            self.body_xyz = np.asarray(self.gps.getValues())
            self.body_speed = np.zeros(3)
        else:
            self.body_speed = (np.asarray(self.gps.getValues()) - self.body_xyz) / (self.timeStep * 1e-3)
            self.body_xyz = np.asarray(self.gps.getValues())

        y = self.body_xyz[1]
        if self.initial_y is None:
            self.initial_y = y

        self.body_rpy = self.inertial_unit.getRollPitchYaw()
        '''
        The roll angle indicates the unit's rotation angle about its x-axis, 
        in the interval [-π,π]. The roll angle is zero when the InertialUnit is horizontal, 
        i.e., when its y-axis has the opposite direction of the gravity (WorldInfo defines 
        the gravity vector).

        The pitch angle indicates the unit's rotation angle about is z-axis, 
        in the interval [-π/2,π/2]. The pitch angle is zero when the InertialUnit is horizontal, 
        i.e., when its y-axis has the opposite direction of the gravity. 
        If the InertialUnit is placed on the Robot with a standard orientation, 
        then the pitch angle is negative when the Robot is going down, 
        and positive when the robot is going up.

        The yaw angle indicates the unit orientation, in the interval [-π,π], 
        with respect to WorldInfo.northDirection. 
        The yaw angle is zero when the InertialUnit's x-axis is aligned with the north direction, 
        it is π/2 when the unit is heading east, and -π/2 when the unit is oriented towards the west. 
        The yaw angle can be used as a compass.
        '''

        more = np.array([
            y - self.initial_y,
            0, 0,
            0.3 * self.body_speed[0], 0.3 * self.body_speed[1], 0.3 * self.body_speed[2],
            # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            self.body_rpy[0] / np.pi, self.body_rpy[1] / (0.5 * np.pi)], dtype=np.float32)

        self.feet_contact = np.zeros(2)
        for j in range(len(self.fsr)):
            fsv = np.asarray(self.fsr[j].getValues())
            '''
            Left Foot Front Left, Left Foot Front Right,
            Left Foot Rear Right, Left Foot Rear Left
            '''
            foot_forces = np.zeros(4)
            foot_forces[0] = fsv[2] / 3.4 + 1.5 * fsv[0] + 1.15 * fsv[1]
            foot_forces[1] = fsv[2] / 3.4 + 1.5 * fsv[0] - 1.15 * fsv[1]
            foot_forces[2] = fsv[2] / 3.4 - 1.5 * fsv[0] - 1.15 * fsv[1]
            foot_forces[3] = fsv[2] / 3.4 - 1.5 * fsv[0] + 1.15 * fsv[1]
            # print('foot_forces: {}'.format(foot_forces))
            if np.min(foot_forces) > 3:
                self.feet_contact[j] = 1

        return np.clip(np.concatenate([more] + [joint_states] + [self.feet_contact]), -5, +5)

    def calc_forward_speed(self):
        '''
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second,
        this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        '''
        direction_r = self.body_xyz - np.asarray(self.boom_base_trans_field.getSFVec3f())
        # print('robot_xyz: {}, boom_base_xyz: {}'.format(self.body_xyz,
        #                                                 np.asarray(self.boom_base_trans_field.getSFVec3f())))
        direction_r = direction_r[[0, 2]] / np.linalg.norm(direction_r[[0, 2]])
        direction_t = np.dot(np.asarray([[0, 1],
                                  [-1, 0]]), direction_r.reshape((-1, 1)))
        return np.dot(self.body_speed[[0, 2]], direction_t)

    def alive_bonus(self, y, pitch):
        return +1 if y > 0.1 and abs(pitch) < 1.0 else -1


    def step(self, action):
        self.apply_action(action)
        simulation_state = self.robot.step(self.timeStep)
        state = self.calc_state()  # also calculates self.joints_at_limit

        # state[0] is body height above ground, body_rpy[1] is pitch
        alive = float(self.alive_bonus(state[0] + self.initial_y,
                                       self.body_rpy[1]))



        progress = self.calc_forward_speed()
        # print('progress: {}'.format(progress))

        feet_collision_cost = 0.0


        '''
        let's assume we have DC motor with controller, and reverse current braking
        '''
        electricity_cost = self.electricity_cost * float(np.abs(
            action * self.joint_speeds).mean())
        electricity_cost += self.stall_torque_cost * float(np.square(action).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        rewards = [
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
        ]

        self.frame += 1

        done = (-1 == simulation_state) or (self._max_episode_steps <= self.frame) \
               or (alive < 0) or (not np.isfinite(state).all())
        # print('frame: {}, alive: {}, done: {}, body_xyz: {}'.format(self.frame, alive, done, self.body_xyz))
        # print('state_{} \n action_{}, reward_{}'.format(state, action, sum(rewards)))
        return state, sum(rewards), done, {}

    def run(self):
        # Main loop.
        for i in range(1000):
            action = np.random.uniform(-1, 1, 6)
            state, reward, done, _ = self.step(action)
            # print('state_{} \n action_{}, reward_{}'.format(state, action, reward))
            if done:
                break

    def reset(self):
        self.initial_y = None
        self.body_xyz = None
        self.joint_angles = None
        self.frame = 0

        for i in range(100):
            for j in self.motors:
                j.setPosition(0)
                self.robot.step(self.timeStep)
        self.robot.simulationResetPhysics()
        self.robot_trans_field.setSFVec3f(self.robot_ini_trans)
        self.robot_rot_field.setSFRotation(self.robot_ini_rot)
        self.boom_body_trans_field.setSFVec3f(self.boom_body_ini_trans)
        self.boom_body_rot_field.setSFRotation(self.boom_body_ini_rot)
        for i in range(10):
            self.robot.step(self.timeStep)
            # print('wait')

        return self.calc_state()