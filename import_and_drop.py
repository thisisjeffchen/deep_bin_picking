import pybullet as p
from time import sleep
import random
import os

physicsClient = p.connect(p.GUI)

p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
urdf_dir = "./meshes/urdf"

NUM = 20
itemIds = []

dirs = os.listdir (urdf_dir)

print dirs
for i in range (NUM):
        idx = random.randint (0, len (dirs) - 1)
        cubeStartPos = [0.5, 0.5 ,2]
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
        d = dirs[idx]
        f = d.split("/")[-1]
        fn = os.path.join (urdf_dir, d, f + ".urdf")
        print "Creating " + f
                
        itemId = p.loadURDF(fn,cubeStartPos, cubeStartOrientation)
        #boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
        cubePos, cubeOrn = p.getBasePositionAndOrientation(itemId)
        print "ItemId: " + str(itemId)
        
        itemIds.append(itemId)

        useRealTimeSimulation = 1

        if (useRealTimeSimulation):
	        p.setRealTimeSimulation(1)

        count = 0
        while count < 100:
	        if (useRealTimeSimulation):
		        p.setGravity(0,0,-10)
		        sleep(0.01) # Time in seconds.
                else:
                        p.stepSimulation()
                count += 1

print "All objects created"
for itemId in itemIds:
        cubePos, cubeOrn = p.getBasePositionAndOrientation(itemId)
        print str(itemId) + "-pos: " + str(cubePos) + "-orn" + str(cubeOrn)
        

raw_input ("Press any key to end...")

                

                

#find objects
#randomly drop objects
#wait for object to settle
#drop another object



