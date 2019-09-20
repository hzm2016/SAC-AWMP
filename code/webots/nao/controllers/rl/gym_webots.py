import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os
from controller import Robot, Supervisor
from roboschool import gym_forward_walker

class Nao(gym.Env):
    """
        Base class for Webots actors in a Scene.
        These environments create single-player scenes and behave like normal Gym environments, if
        you don't use multiplayer.
        """

    electricity_cost = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    joints_at_limit_cost = -0.2  # discourage stuck joints

    def __init__(self, action_dim, obs_dim):
        self.robot = Supervisor()
        self.robot_Node = self.robot.getFromDef('Nao')
        self.timeStep = int(self.robot.getBasicTimeStep()) # ms
        self.find_and_enable_devices()

        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        high = np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.reset()

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

        # leg pitch motors
        self.shoulderMotor = [self.robot.getMotor('RShoulderPitch'),
                              self.robot.getMotor('LShoulderPitch')]

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
            j.setTorque(j.getMaxTorque() * float(np.clip(a[n], -1, +1)))


    def calc_state(self):
        joint_states = np.zeros(2*len(self.legPitchMotor))
        # even elements [0::2] position, scaled to -1..+1 between limits
        for r in range(6):
            joint_angle = self.legPitchSensor[r].getValue()
            max_joint_angle = self.legPitchMotor[r].getMaxPosition()
            min_joint_angle = self.legPitchMotor[r].getMinPosition()
            print('max_q_{}, min_q_{}'.format(max_joint_angle, min_joint_angle))
            joint_states[2 * r] = -(joint_angle - 0.5 * (max_joint_angle + min_joint_angle)) \
                              / (0.5 * (max_joint_angle - min_joint_angle))
            if r in [1, 4]: # only the direction of the knee is the same as human
                joint_states[r] = -joint_states[r]
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        for r in range(6):
            joint_states[2 * r + 1] = self.legPitchMotor[r].getVelocity() \
                              / abs(self.legPitchMotor[r].getMaxVelocity())

        self.joint_speeds = joint_states[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(joint_states[0::2]) > 0.99)

        if self.body_xyz is None:
            self.body_xyz = self.gps.getValues()
            self.body_speed = np.zeros(3)
        else:
            self.body_speed = (self.gps.getValues() - self.body_xyz)/ (self.timeStep * 1e-3)
            self.body_xyz = self.gps.getValues()

        z = self.body_xyz[2]
        if self.initial_z is None:
            self.initial_z = z

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
            z - self.initial_z,
            0, 0,
            0.3 * self.body_speed[0], 0.3 * self.body_speed[1], 0.3 * self.body_speed[2],
            # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            self.body_rpy[0] / np.pi, self.body_rpy[1] / (0.5 * np.pi)], dtype=np.float32)

        self.feet_contact = np.zeros(2)
        for j in range(len(self.fsr)):
            foot_forces = self.fsr[j].getValues()
            if abs(foot_forces[2]) > 3:
                self.feet_contact[j] = 1

        return np.clip(np.concatenate([more] + [joint_states] + [self.feet_contact]), -5, +5)

    # def calc_potential(self):
    #     # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
    #     # all rewards have rew/frame units and close to 1.0
    #     return - self.walk_target_dist / self.scene.dt

    def step(self, a):
        done = False
        reward = 0
        # self.apply_action(a)
        if -1 == self.robot.step(self.timeStep):
            done = True
        state = self.calc_state()  # also calculates self.joints_at_limit

        # alive = float(self.alive_bonus(state[0] + self.initial_z,
        #                                self.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        # done = alive < 0
        # if not np.isfinite(state).all():
        #     print("~INF~", state)
        #     done = True
        #
        # potential_old = self.potential
        # self.potential = self.calc_potential()
        # progress = float(self.potential - potential_old)
        #
        # feet_collision_cost = 0.0
        # for i, f in enumerate(self.feet):
        #     contact_names = set(x.name for x in f.contact_list())
        #     # print("CONTACT OF '%s' WITH %s" % (f.name, ",".join(contact_names)) )
        #     self.feet_contact[i] = 1.0 if (self.foot_ground_object_names & contact_names) else 0.0
        #     if contact_names - self.foot_ground_object_names:
        #         feet_collision_cost += self.foot_collision_cost
        #
        # electricity_cost = self.electricity_cost * float(np.abs(
        #     a * self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        # electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        #
        # joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)
        #
        # self.rewards = [
        #     alive,
        #     progress,
        #     electricity_cost,
        #     joints_at_limit_cost,
        #     feet_collision_cost
        # ]
        #
        # self.frame += 1
        # if (done and not self.done) or self.frame == self.spec.max_episode_steps:
        #     self.episode_over(self.frame)
        # self.done += done  # 2 == 1+True
        # self.reward += sum(self.rewards)
        # self.HUD(state, a, done)
        return state, reward, bool(done), {}

    def run(self):
        # Main loop.
        for i in range(1000):
            action = 0.1 * np.random.uniform(0, 0, 6)
            state, reward, done, _ = self.step(action)
            if done:
                break
            print('state_{}, action_{}'.format(state, action))
            # print('gps: {}, inertial_unit: {}'.format(self.gps.getValues(),
            #                                           self.inertial_unit.getRollPitchYaw()))
            # for j in range(len(self.fsr)):
            #     print('fsr: {}_{}'.format(j, self.fsr[j].getValues()))
            #
            # for j in range(len(self.legPitchMotor)):
            #     print('legPitch: {}_{}'.format(j, self.legPitchMotor[j].getVelocity()))
            #
            # for j in range(len(self.legPitchSensor)):
            #     print('legPitch: {}_{}'.format(j, self.legPitchSensor[j].getValue()))
            #
            # for j in range(len(self.shoulderMotor)):
            #     self.shoulderMotor[j].setPosition(1.57)

    def reset(self):
        # self.robot.simulationReset()
        self.initial_z = None
        self.body_xyz = None