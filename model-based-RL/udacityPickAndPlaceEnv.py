import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import udacityKuka
import random
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class UdacityPickAndPlaceEnv(gym.Env):
  metadata = {
 #     'render.modes': ['human', 'rgb_array'],
      'render.modes': ['human',],
      'video.frames_per_second' : 50
  }

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps = 100000):
    #print("KukaGymEnv __init__")
    self._isDiscrete = isDiscrete
    self._timeStep = 1./240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40

    self._p = p
    if self._renders:



      cid = p.connect(p.SHARED_MEMORY, options='--background_color_red=0.93 --background_color_green=0.97 --background_color_blue=0.97')
      if (cid<0):
         cid = p.connect(p.GUI, options='--background_color_red=0.93 --background_color_green=0.97 --background_color_blue=0.97')
      p.resetDebugVisualizerCamera(1.3,180,-41,[0.52,-0.2,-0.33])
    else:
      p.connect(p.DIRECT, options='--background_color_red=0.93 --background_color_green=0.97 --background_color_blue=0.97')
    #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    self.seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())
    #print("observationDim")
    #print(observationDim)

    observation_high = np.array([largeValObservation] * observationDim)
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
       action_dim = 3
       self._action_bound = 1
       action_high = np.array([self._action_bound] * action_dim)
       self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None

  def reset(self):
    #print("KukaGymEnv _reset")
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
  

    p.loadURDF("pybullet_models/table.urdf", 0.5000000,0.00000,-.770000,0.000000,0.000000,0.0,1.0)



    ang = 3.14*0.5+3.1415925438*random.random()



    x = np.random.uniform(low=0, high=1)
    y = np.random.uniform(low=0, high=1)

    z = 0.18693777120558974

    y0 = -0.34350241525285494
    y1 = 0.34350241525285494

    x0 = 0.2648612700344569
    x1 = 0.7048612700344569

    x_real = x *(x1 - x0) + x0
    y_real = y *(y1 - y0) + y0

    orn = p.getQuaternionFromEuler([0,0,ang])
    self.blockUid =p.loadURDF("pybullet_models/block.urdf", [x_real,y_real, 0.18693777120558974],[orn[0],orn[1],orn[2],orn[3]], useFixedBase=True)

    p.setGravity(0,0,-10)
    self._kuka = udacityKuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

    return np.array(self._observation)

  def __del__(self):
    p.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):


    blockPos,blockOrn=p.getBasePositionAndOrientation(self.blockUid)
    linear_velocity, angular_velocity = p.getBaseVelocity(self.blockUid)

    reward = -1000

    state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaGripperIndex)
    pos = state[0]

    return np.append(np.append(np.array(pos),  np.array(blockPos)), np.array(linear_velocity))


    return self._observation

  def step(self, action):

    return self.step2( action)

  def step2(self, action):
    '''
    action is a 2D vector (target x,y location of gripper)

    '''
    for i in range(self._actionRepeat):
      self._kuka.applyAction(action)
      p.stepSimulation()
      if self._termination():
        break
      self._envStepCounter += 1
    if self._renders:
      time.sleep(self._timeStep)
    self._observation = self.getExtendedObservation()

    #print("self._envStepCounter")
    #print(self._envStepCounter)

    done = self._termination()

    #print("actionCost")
    #print(actionCost)
    reward = self._reward()
    #print("reward")
    #print(reward)

    #print("len=%r" % len(self._observation))

    return np.array(self._observation), reward, done, {}

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])

    base_pos,orn = self._p.getBasePositionAndOrientation(self._kuka.kukaUid)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
        nearVal=0.1, farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        #renderer=self._p.ER_TINY_RENDERER)


    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

    rgb_array = rgb_array[:, :, :3]
    return rgb_array


  def _termination(self):
    #print (self._kuka.endEffectorPos[2])
    state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]

    #print("self._envStepCounter")
    #print(self._envStepCounter)
    if (self.terminated or self._envStepCounter>self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.005



    return False

  def _reward(self):

    #rewards is height of target object
    blockPos,blockOrn=p.getBasePositionAndOrientation(self.blockUid)

    reward = -1000

    state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaGripperIndex)
    pos = state[0]

    reward = np.linalg.norm(np.array(pos) - np.array(blockPos))

    return -reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
