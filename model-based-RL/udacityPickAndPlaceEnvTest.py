#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

from udacityPickAndPlaceEnv import UdacityPickAndPlaceEnv
import time

from pymouse import PyMouse
from pykeyboard import PyKeyboard

m = PyMouse()

def main():

	environment = UdacityPickAndPlaceEnv(renders=True,isDiscrete=False, maxSteps = 10000000)
	
	motorsIds=[]
	
	dv = 0.01 
	motorsIds.append(environment._p.addUserDebugParameter("posX",-dv,dv,0))
	motorsIds.append(environment._p.addUserDebugParameter("posY",-dv,dv,0))
	motorsIds.append(environment._p.addUserDebugParameter("posZ",-dv,dv,0))
	motorsIds.append(environment._p.addUserDebugParameter("yaw",-dv,dv,0))
	motorsIds.append(environment._p.addUserDebugParameter("fingerAngle",0,0.3,.3))
	
	done = False
	while (not done):
	     
		action=[]
		for motorId in motorsIds:
			action.append(environment._p.readUserDebugParameter(motorId))
			action = [0.2648612700344569, -0.34350241525285494]

		x_dim, y_dim = m.screen_size()
		x_mouse, y_mouse = m.position()

		action = [x_mouse/x_dim,y_mouse/y_dim]

		state, reward, done, info = environment.step2(action)
		obs = environment.getExtendedObservation()
		print(obs)

if __name__=="__main__":
    main()
