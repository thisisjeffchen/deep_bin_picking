"""Demo script to run a simulation."""

import logging
import math
import os
import random

import numpy as np

import pybullet as pb

logging.basicConfig(level=logging.INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_DIR = os.path.join(ROOT_DIR, 'meshes', 'urdf')
GRIPPER_DIR = os.path.join(ROOT_DIR, 'meshes', 'grippers')
GRIPPER = 'yumi_metal_spline'


class Scene(object):
    """Manages a scene and its contents."""

    def __init__(self, client_mode=pb.GUI, gravity=[0, 0, -20], timestep=0.0001,
                 crate_wall_thickness=0.002, crate_wall_width=0.2, crate_wall_height=0.15):
        """Set up scene parameters and fixed models."""
        self.client_mode = client_mode
        self.client_id = pb.connect(client_mode)
        self.gravity = gravity
        self.timestep = timestep
        self.crate_wall_thickness = crate_wall_thickness
        self.crate_wall_width = crate_wall_width
        self.crate_wall_height = crate_wall_height

        # Simulation
        pb.setGravity(*gravity, physicsClientId=self.client_id)
        self.step = 0
        if timestep is not None:
            pb.setTimeStep(timestep, physicsClientId=self.client_id)

        # Scene
        self.crate_side_ids = []
        self._add_crate()

        # Gripper
        self.gripper_id = None

        # Items
        self.item_ids = []

    def add_item(self, name, position, orientation):
        """Add the specified item in the specified pose."""
        logging.info('Adding item "{}"'.format(name))
        file_path = os.path.join(URDF_DIR, name, name + '.urdf')
        item_id = pb.loadURDF(
            file_path, position, orientation, physicsClientId=self.client_id
        )
        logging.info('Item assigned object id {}'.format(item_id))
        self.item_ids.append(item_id)
        (position, orientation) = pb.getBasePositionAndOrientation(
            item_id, physicsClientId=self.client_id
        )
        return (item_id, position, orientation)

    def get_item_pose(self, item_id, to_euler=False):
        """Return the pose of the specified item."""
        (position, orientation) = pb.getBasePositionAndOrientation(
            item_id, physicsClientId=self.client_id
        )
        if to_euler:
            orientation = pb.getEulerFromQuaternion(orientation)
        return (position, orientation)

    def simulate(self, steps=None):
        """Simulate physics for the specified number of steps."""
        initial_step = self.step
        while steps is None or self.step - initial_step < steps:
            pb.stepSimulation(physicsClientId=self.client_id)
            self.step += 1

    def _add_gripper(self):
        """Add a gripper. Currently broken."""
        logging.info('Adding gripper')
        file_path = os.path.join(GRIPPER_DIR, GRIPPER, 'base.obj')
        cube_start_orientation = pb.getQuaternionFromEuler([0, 0, 0])
        self.gripper_id = pb.loadURDF(
            file_path, [0, 0, 3], cube_start_orientation,
            physicsClientId=self.client_id
        )
        logging.info('Gripper assigned id {}'.format(self.gripper_id))
        pos, ori = pb.getBasePositionAndOrientation(
            self.gripper_id, physicsClientId=self.client_id
        )
        return self.gripper_id, pos, ori

    def _add_crate_wall(self, side):
        """Add a crate wall in the specified side (0-3)."""
        if side % 2 == 0:
            x_length = self.crate_wall_thickness / 2
            y_length = self.crate_wall_width / 2
            x_center = self.crate_wall_width / 2
            y_center = 0
        else:
            x_length = self.crate_wall_width / 2
            y_length = self.crate_wall_thickness / 2
            x_center = 0
            y_center = self.crate_wall_width / 2
        if side > 1:
            x_center *= -1
            y_center *= -1
        z_length = self.crate_wall_height / 2
        z_center = z_length
        collision_shape_id = pb.createCollisionShape(
            pb.GEOM_BOX, halfExtents=[x_length, y_length, z_length],
            collisionFramePosition=[x_center, y_center, z_center],
            physicsClientId=self.client_id
        )
        visual_shape_id = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=[x_length, y_length, z_length],
            visualFramePosition=[x_center, y_center, z_center],
            physicsClientId=self.client_id

        )
        multibody_id = pb.createMultiBody(
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            physicsClientId=self.client_id
        )
        self.crate_side_ids.append(multibody_id)

    def _add_crate_floor(self):
        """Add a crate floor."""
        x_length = self.crate_wall_width / 2
        y_length = self.crate_wall_width / 2
        z_length = self.crate_wall_thickness / 2
        collision_shape_id = pb.createCollisionShape(
            pb.GEOM_BOX, halfExtents=[x_length, y_length, z_length],
            collisionFramePosition=[0, 0, 0], physicsClientId=self.client_id
        )
        visual_shape_id = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=[x_length, y_length, z_length],
            visualFramePosition=[0, 0, 0], physicsClientId=self.client_id

        )
        multibody_id = pb.createMultiBody(
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            physicsClientId=self.client_id
        )
        self.crate_side_ids.append(multibody_id)

    def _add_crate(self):
        """Add a complete crate."""
        print('Creating crate')
        self._add_crate_floor()
        for side in range(4):
            self._add_crate_wall(side)


class ScenePopulator(object):
    """Stochastically populates a Scene with items."""

    def __init__(
            self, scene, min_items=15, max_items=25,
            mean_position=np.array([0, 0, 0.25]), position_ranges=np.array([0.1, 0.1, 0.1]),
            sim_height_threshold=0.15, sim_check_interval=1000
    ):
        """Load the database of item models from the filesystem."""
        self.scene = scene
        self.sim_height_threshold = sim_height_threshold
        self.sim_check_interval = sim_check_interval

        # Random distribution parameters
        self.min_items = min_items
        self.max_items = max_items
        self.mean_position = mean_position
        self.position_ranges = position_ranges

        # Item loading
        self.full_item_database = sorted(os.listdir(URDF_DIR))
        self.excluded_items = {
            # Invalid objects
            '31340e691294a750d30ee0b5a9888c0b', '38dd2a8d2c984e2b6c1cd53dbc9f7b8e',
            '3c80c41399d2c92334fb047a3245866d', '3f497f8d7dd8922a57e59dddfaae7826',
            '4fcd289d3e82bb08588f04a271bfa5eb', '4fcd289d3e82bb08588f04a271bfa5eb',
            '68582543c4c6d0bccfdfe3f21f42a111', '9a52843cc89cd208362be90aaa182ec6',
            'a4584986b4baf1479a2840934f7f84bc', 'a86d587f38569fdf394a7890920ef7fd',
            'bacef3777b91b02f29c5890b07f3a65', 'c09e3db27668639a69fba573ec0b31f3',
            'c453274b341f8c4ec2b9bcaf66ea9919', 'dc0c4db824981b8cf29c5890b07f3a65',
            'pliers_standard',
            # Annoying objects
            'f44301a26ca7f57c70d5704d62bc4186'
        }
        self.item_database = sorted(list(set(
            self.full_item_database) - self.excluded_items
        ))
        logging.info('Available items: {}'.format(self.item_database))

    def _sample_num_items(self):
        """Sample a uniform random distribution for the number of items to add."""
        return random.randint(self.min_items, self.max_items)

    def _sample_item(self):
        """Sample a uniform random distribution for the item to add."""
        return random.choice(self.item_database)

    def _sample_position(self):
        """Sample a uniform random distribution for the position of the item to add."""
        return np.random.uniform(self.mean_position - self.position_ranges / 2,
                                 self.mean_position + self.position_ranges / 2)

    def _sample_orientation(self):
        """Sample a uniform random distribution for the orientation of the item to add."""
        euler = [random.uniform(0, 2 * math.pi) for _ in range(3)]
        return pb.getQuaternionFromEuler(euler)

    def add_item(self):
        """Add a random item to the scene."""
        return self.scene.add_item(
            self._sample_item(), self._sample_position(), self._sample_orientation()
        )

    def simulate_to(self, item_id, height_threshold, check_interval=1000):
        """Simumlate the scene until the specified item has fallen below the specified height."""
        while True:
            self.scene.simulate(steps=check_interval)
            pose = self.scene.get_item_pose(item_id)
            if pose[0][2] < height_threshold:
                return

    def add_items(self, num_items=None):
        """Stochastically populate a scene by randomly dropping items into a crate."""
        if num_items is None:
            num_items = self._sample_num_items()
        for i in range(num_items):
            (item_id, _, _) = self.add_item()
            self.simulate_to(
                item_id, self.sim_height_threshold, self.sim_check_interval
            )


def main():
    """Initialize and populate a random scene and simulate it."""
    scene = Scene()
    populator = ScenePopulator(scene)
    populator.add_items(num_items=5)
    print('All items created!')

    for item_id in scene.item_ids:
        (position, orientation) = scene.get_item_pose(item_id)
        print(str(item_id) + '-pos: ' + str(position) + '-orn' + str(orientation))

    scene.simulate()
    input('Press any key to end...')


if __name__ == '__main__':
    main()
