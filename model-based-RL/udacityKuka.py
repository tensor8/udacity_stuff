import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data


class Kuka:

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.maxVelocity = .35
    self.maxForce = 200.
    self.fingerAForce = 2 
    self.fingerBForce = 2.5
    self.fingerTipForce = 2
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace =21
    self.useOrientation = 1
    self.kukaEndEffectorIndex = 6
    self.kukaGripperIndex = 7
    #lower limits for null space
    self.ll=[-.967,-2 ,-2.96,0.19,-2.96,-2.09,-3.05]
    #upper limits for null space
    self.ul=[.967,2 ,2.96,2.29,2.96,2.09,3.05]
    #joint ranges for null space
    self.jr=[5.8,4,5.8,4,5.8,4,6]
    #restposes for null space
    self.rp=[0,0,0,0.5*math.pi,0,-math.pi*0.5*0.66,0]
    #joint damping coefficents
    self.jd=[0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001]
    self.reset()
    
  def reset(self):
    objects = p.loadSDF(os.path.join(self.urdfRootPath,"kuka_iiwa/kuka_with_gripper2.sdf"))
    self.kukaUid = objects[0]

    p.resetBasePositionAndOrientation(self.kukaUid,[-0.100000,0.000000,0.070000],[0.000000,0.000000,0.000000,1.000000])
    self.jointPositions=[ 0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048, -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200 ]
    self.numJoints = p.getNumJoints(self.kukaUid)
    for jointIndex in range (self.numJoints):
      p.resetJointState(self.kukaUid,jointIndex,self.jointPositions[jointIndex])
      #p.setJointMotorControl2(self.kukaUid,jointIndex,p.POSITION_CONTROL,0,0)
    
    self.endEffectorPos = [0.537,0.0,0.5]
    self.endEffectorAngle = 0
    
    
    self.motorNames = []
    self.motorIndices = []

    import random
    xpos = 0.55 +0.12*random.random()
    ypos = 0 +0.2*random.random()
    ang = 3.14*0.5+3.1415925438*random.random()
    orn = p.getQuaternionFromEuler([0,0,ang])
    orn = np.array((-0.0, 1, 0.0, 0.0))
    pos = (0.5321085918022986, -0.0011177175302174629, 0.2522914797947443),
    self.constraint = p.createConstraint(self.kukaUid,7,-1,-1,p.JOINT_FIXED,pos,orn,[0.500000,0.300006,0.700000])

  def getActionDimension(self):
    if (self.useInverseKinematics):
      return len(self.motorIndices)
    return 6 #position x,y,z and roll/pitch/yaw euler angles of end effector

  def getObservationDimension(self):
    return len(self.getObservation())

  def getObservation(self):
    observation = []
    state = p.getLinkState(self.kukaUid,self.kukaGripperIndex)
    pos = state[0]
    orn = state[1]
    euler = p.getEulerFromQuaternion(orn)
        
    observation.extend(list(pos))
    observation.extend(list(euler))
    
    return observation

  def applyAction(self, motorCommands):
    #p.applyExternalForce(self.kukaUid,7, [1000,0,0], [0,0,0], p.WORLD_FRAME)
    '''
    x in [0,1]
    y in [0,1]

    '''
    x = motorCommands[0]
    y = motorCommands[1]

    z = 0.18693777120558974

    y0 = -0.34350241525285494
    y1 = 0.34350241525285494

    x0 = 0.2648612700344569
    x1 = 0.7048612700344569

    x_real = x *(x1 - x0) + x0
    y_real = y *(y1 - y0) + y0


    orn = np.array((-3.999367970686227e-06, 0.9999999999838405, 6.967182601629827e-07, 3.979782852686303e-06))

    pos = (x_real,y_real,z)

    p.changeConstraint(self.constraint, pos , orn, maxForce=10000)
