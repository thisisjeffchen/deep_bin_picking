import pybullet as p
from time import sleep
import random

physicsClient = p.connect(p.GUI)

p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")

NUM = 2

for i in range (NUM):
        cubeStartPos = [random.randint(0,2),random.randint (0,2),10]
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
        boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
        cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
        print "Making robot"

p.loadURDF("plane.urdf",[0,0,-0.3],useFixedBase=True)
kukaId = p.loadURDF("kuka_iiwa/model.urdf",[0,0,0],useFixedBase=True)
p.resetBasePositionAndOrientation(kukaId,[0,0,0],[0,0,0,1])
kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(kukaId)
if (numJoints!=7):
	exit()



useRealTimeSimulation = 0

if (useRealTimeSimulation):
	p.setRealTimeSimulation(1)

while 1:
	if (useRealTimeSimulation):
		p.setGravity(0,0,-10)
		sleep(0.01) # Time in seconds.
	else:
		p.stepSimulation()
