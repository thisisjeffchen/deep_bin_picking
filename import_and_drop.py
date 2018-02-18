"""Demo script to run a simulation."""

import logging
import os
import random
from time import sleep

import pybullet as p

logging.basicConfig(level=logging.INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_DIR = os.path.join(ROOT_DIR, 'meshes', 'urdf')

GRAVITY = [0, 0, -10]

physicsClient = p.connect(p.GUI)

class Scene(object):
    def __init__(self):
        self.plane_id = None
        self.item_ids = []
        p.setGravity(*GRAVITY)
        self.plane_id = p.loadURDF('plane.urdf')

    def add_object(self, name, position, orientation):
        logging.info('Adding object "{}"'.format(name))
        file_path = os.path.join(URDF_DIR, name, name + '.urdf')
        item_id = p.loadURDF(file_path, position, orientation)
        logging.info('Object assigned id {}'.format(item_id))
        self.item_ids.append(item_id)
        (position, orientation) = p.getBasePositionAndOrientation(item_id)
        return (item_id, position, orientation)

class ScenePopulator(object):
    def __init__(self, scene, mean_position, mean_orientation):
        self.scene = scene
        self.mean_position = mean_position
        self.mean_orientation = mean_orientation
        self.object_database = os.listdir(URDF_DIR)
        logging.info('Available objects: {}'.format(self.object_database))

    def sample_position(self):
        return self.mean_position

    def sample_orientation(self):
        return p.getQuaternionFromEuler(self.mean_orientation)

    def sample_object(self):
        return random.choice(self.object_database)

    def add_object(self):
        start_position = self.sample_position()
        start_orientation = self.sample_orientation()
        object_name = self.sample_object()
        return self.scene.add_object(object_name, start_position, start_orientation)

NUM_OBJECTS = 20
scene = Scene()
populator = ScenePopulator(scene, [0.5, 0.5, 2], [0, 0, 0])

for i in range(NUM_OBJECTS):
    (item_id, _, _) = populator.add_object()

    useRealTimeSimulation = True
    if (useRealTimeSimulation):
        p.setRealTimeSimulation(1)

    count = 0
    while count < 100:
        if (useRealTimeSimulation):
            p.setGravity(*GRAVITY)
            sleep(0.01)  # Time in seconds.
        else:
            p.stepSimulation()
            count += 1

print('All objects created')
for itemId in scene.item_ids:
    cubePos, cubeOrn = p.getBasePositionAndOrientation(itemId)
    print(str(itemId) + '-pos: ' + str(cubePos) + '-orn' + str(cubeOrn))


input('Press any key to end...')


# find objects
# randomly drop objects
# wait for object to settle
# drop another object
