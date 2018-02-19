"""Demo script to run a simulation."""

import logging
import math
import os
import random
from time import sleep

import numpy as np
import pybullet as p

logging.basicConfig(level=logging.INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_DIR = os.path.join(ROOT_DIR, 'meshes', 'urdf')

physicsClient = p.connect(p.GUI)  # Change p.GUI to p.DIRECT to run headlessly.

class Scene(object):
    """Manages a scene and its contents."""
    def __init__(self, gravity=[0, 0, -20], timestep=0.0001):
        self.gravity = gravity
        self.timestep = timestep

        # Simulation
        p.setGravity(*gravity)
        self.step = 0
        if timestep is not None:
            p.setTimeStep(timestep)

        # Items
        self.plane_id = None
        self.item_ids = []
        self.plane_id = p.loadURDF('plane.urdf')
        self.tray_id = p.loadURDF('tray/traybox.urdf', globalScaling=0.5)

    def add_object(self, name, position, orientation):
        logging.info('Adding object "{}"'.format(name))
        file_path = os.path.join(URDF_DIR, name, name + '.urdf')
        item_id = p.loadURDF(file_path, position, orientation)
        logging.info('Object assigned id {}'.format(item_id))
        self.item_ids.append(item_id)
        (position, orientation) = p.getBasePositionAndOrientation(item_id)
        return (item_id, position, orientation)

    def simulate(self, steps=None):
        initial_step = self.step
        while steps is None or self.step - initial_step < steps:
            p.stepSimulation()
            self.step += 1

class ScenePopulator(object):
    """Stochastically populates a Scene with objects."""
    def __init__(self, scene, min_objects=15, max_objects=25,
                 mean_position=np.array([0, 0, 0.25]),
                 position_ranges=np.array([0.1, 0.1, 0.1])):
        self.scene = scene

        # Random distribution parameters
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.mean_position = mean_position
        self.position_ranges = position_ranges

        # Object loading
        self.full_object_database = sorted(os.listdir(URDF_DIR))
        self.excluded_objects = {
            '31340e691294a750d30ee0b5a9888c0b', '38dd2a8d2c984e2b6c1cd53dbc9f7b8e',
            '3c80c41399d2c92334fb047a3245866d', '3f497f8d7dd8922a57e59dddfaae7826',
            '4fcd289d3e82bb08588f04a271bfa5eb', '4fcd289d3e82bb08588f04a271bfa5eb',
            '68582543c4c6d0bccfdfe3f21f42a111', '9a52843cc89cd208362be90aaa182ec6',
            'a4584986b4baf1479a2840934f7f84bc', 'a86d587f38569fdf394a7890920ef7fd',
            'bacef3777b91b02f29c5890b07f3a65', 'c09e3db27668639a69fba573ec0b31f3',
            'c453274b341f8c4ec2b9bcaf66ea9919', 'dc0c4db824981b8cf29c5890b07f3a65',
            'pliers_standard'
        }
        self.object_database = sorted(list(set(
            self.full_object_database) - self.excluded_objects
        ))
        logging.info('Available objects: {}'.format(self.object_database))

    def sample_num_objects(self):
        return random.randint(self.min_objects, self.max_objects)

    def sample_object(self):
        return random.choice(self.object_database)

    def sample_position(self):
        return np.random.uniform(self.mean_position - self.position_ranges / 2,
                                 self.mean_position + self.position_ranges / 2)

    def sample_orientation(self):
        return p.getQuaternionFromEuler([
            random.uniform(0, 2 * math.pi) for _ in range(3)
        ])

    def add_object(self):
        return self.scene.add_object(
            self.sample_object(), self.sample_position(), self.sample_orientation()
        )

    def simulate_to(self, object_id, height_threshold):
        while True:
            self.scene.simulate(steps=1000)
            if p.getBasePositionAndOrientation(object_id)[0][2] < height_threshold:
                return

    def add_objects(self, num_objects=None, height_threshold=0.1):
        if num_objects is None:
            num_objects = self.sample_num_objects()
        for i in range(num_objects):
            (object_id, _, _) = populator.add_object()
            populator.simulate_to(object_id, height_threshold)


scene = Scene()
populator = ScenePopulator(scene)
populator.add_objects()
#populator.add_gripper ()

print('All objects created!')

for item_id in scene.item_ids:
    cubePos, cubeOrn = p.getBasePositionAndOrientation(item_id)
    print(str(item_id) + '-pos: ' + str(cubePos) + '-orn' + str(cubeOrn))


scene.simulate()
input('Press any key to end...')


# find objects
# randomly drop objects
# wait for object to settle
# drop another object
