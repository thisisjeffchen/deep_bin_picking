"""Management of PyBullet scenes."""

import logging
import math
import os
import random

import numpy as np

import pybullet as pb

# Import input/raw-input with python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

logging.basicConfig(level=logging.INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_DIR = os.path.join(ROOT_DIR, 'meshes', 'urdf')
GRIPPER_DIR = os.path.join(ROOT_DIR, 'meshes', 'grippers')
GRIPPER = 'yumi_metal_spline'


def compute_pose_deltas(poses, prev_poses):
    """Given two dicts of item poses, computes position and orientation deltas for each item."""
    shared_items = set(poses.keys()) & set(prev_poses.keys())
    position_deltas = {
        item_id: np.array(poses[item_id][0]) - np.array(prev_poses[item_id][0])
        for item_id in shared_items
    }
    angle_deltas = {
        item_id: np.array(poses[item_id][1]) - np.array(prev_poses[item_id][1])
        for item_id in shared_items
    }
    return (position_deltas, angle_deltas)


class Scene(object):
    """Manages a scene and its contents."""

    def __init__(
            self, show_gui=False, gravity=[0, 0, -40], timestep_interval=0.0001,
            crate_wall_thickness=0.002, crate_wall_width=0.15, crate_wall_height=0.15,
            bounds_height_thresholds=(-0.1, 0.35)
    ):
        """Set up scene parameters and fixed models."""
        self.show_gui = show_gui
        self.client_mode = pb.GUI if self.show_gui else pb.DIRECT
        self.client_id = pb.connect(self.client_mode)
        self.gravity = gravity
        self.timestep_interval = timestep_interval
        self.crate_wall_thickness = crate_wall_thickness
        self.crate_wall_width = crate_wall_width
        self.crate_wall_height = crate_wall_height
        self.bounds_height_thresholds = bounds_height_thresholds

        # Simulation
        pb.setGravity(*gravity, physicsClientId=self.client_id)
        self.step = 0
        pb.setTimeStep(timestep_interval, physicsClientId=self.client_id)

        # Scene
        self.crate_side_ids = []
        self._add_crate()

        # Gripper
        self.gripper_id = None

        # Items
        self.item_ids = {}
        self.item_names = {}
        self.linear_velocity_clamps = {}
        self.angular_velocity_clamps = {}

    def add_item(self, name, position, orientation, lateral_friction=None,
                 spinning_friction=None, rolling_friction=None):
        """Add the specified item in the specified pose."""
        logging.info('Adding item "{}"'.format(name))
        print "Adding"
        print name
        file_path = os.path.join(URDF_DIR, name, name + '.urdf')
        item_id = pb.loadURDF(file_path, position, orientation,
                              physicsClientId=self.client_id)
        for link_id in range(-1, pb.getNumJoints(item_id, physicsClientId=self.client_id)):
            if lateral_friction is not None:
                pb.changeDynamics(item_id, link_id, lateralFriction=lateral_friction)
            if spinning_friction is not None:
                pb.changeDynamics(item_id, link_id, spinningFriction=spinning_friction)
            if rolling_friction is not None:
                pb.changeDynamics(item_id, link_id, rollingFriction=rolling_friction)
        logging.info('Item assigned object id {}'.format(item_id))
        self.item_ids[name] = item_id
        self.item_names[item_id] = name
        (position, orientation) = pb.getBasePositionAndOrientation(
            item_id, physicsClientId=self.client_id
        )
        return (item_id, position, orientation)

    def get_pose(self, object_id, to_euler=False):
        """Return the pose of the specified item as numpy arrays."""
        (position, orientation) = pb.getBasePositionAndOrientation(
            object_id, physicsClientId=self.client_id
        )
        if to_euler:
            orientation = pb.getEulerFromQuaternion(orientation)
        return (np.array(position), np.array(orientation))

    def get_item_poses(self, to_euler=False, as_vector=False):
        """Return the poses of all items as numpy arrays."""
        return {item_id: self.get_pose(item_id, to_euler)
                for item_id in self.item_ids.values()}

    def clamp_item_velocity(self, item_id, linear=None, angular=None):
        """Add a velocity clamp to move an item at constant velocity during simumlation."""
        if linear is not None:
            self.linear_velocity_clamps[item_id] = linear
        if angular is not None:
            self.angular_velocity_clamps[item_id] = angular

    def unclamp_item_velocity(self, item_id):
        """Remove any velocity clamp on an item."""
        return (self.linear_velocity_clamps.pop(item_id, None),
                self.angular_velocity_clamps.pop(item_id, None))

    def remove_bounds_items(self):
        """Remove items out of height bounds."""
        item_poses = self.get_item_poses(as_vector=False)
        item_heights = {id: pose[0][2] for (id, pose) in item_poses.items()}
        bounds_items = {
            id for (id, height) in item_heights.items()
            if (height < self.bounds_height_thresholds[0]
                or height > self.bounds_height_thresholds[1])
        }
        for item_id in bounds_items:
            self.remove_item(item_id)
        return bounds_items

    def remove_item(self, item_id):
        """Clear the scene of items to reset the scene."""
        pb.removeBody(item_id, physicsClientId=self.client_id)
        name = self.item_names.pop(item_id)
        self.item_ids.pop(name)
        self.unclamp_item_velocity(item_id)

    def remove_all_items(self):
        """Clear the scene of items to reset the scene."""
        for item_id in self.item_ids.values():
            pb.removeBody(item_id, physicsClientId=self.client_id)
        self.item_ids.clear()
        self.item_names.clear()
        self.linear_velocity_clamps.clear()
        self.angular_velocity_clamps.clear()

    def simulate(self, steps=None, velocity_clamps=True):
        """Simulate physics for the specified number of steps."""
        initial_step = self.step
        while steps is None or self.step - initial_step < steps:
            if velocity_clamps:
                for (item, velocity) in self.linear_velocity_clamps.items():
                    pb.resetBaseVelocity(item, linearVelocity=velocity,
                                         physicsClientId=self.client_id)
                for (item, velocity) in self.angular_velocity_clamps.items():
                    pb.resetBaseVelocity(item, angularVelocity=velocity,
                                         physicsClientId=self.client_id)
            pb.stepSimulation(physicsClientId=self.client_id)
            self.step += 1
        return self.step - initial_step

    def simulate_to_steady_state(self, position_delta_threshold, angle_delta_threshold,
                                 check_interval=0.02):
        """Simulate physics until items have stopped moving."""
        num_timesteps = int(check_interval / self.timestep_interval)
        prev_poses = None
        poses = None
        initial_step = self.step
        bounds_removed_items = set()
        while True:
            # Update poses
            prev_poses = poses
            poses = self.get_item_poses(to_euler=False, as_vector=True)

            # Simulate
            self.simulate(steps=num_timesteps)
            removed_items = self.remove_bounds_items()
            if removed_items:
                logging.info('Removed items out of simulation bounds: {}'
                             .format(removed_items))
                bounds_removed_items |= removed_items

            # Check for steady state
            if not self.item_ids:
                break
            if prev_poses is None:
                continue
            (position_deltas, angle_deltas) = compute_pose_deltas(poses, prev_poses)
            max_position_delta = np.max(np.abs(np.array(list(position_deltas.values()))))
            max_angle_delta = np.max(np.abs(np.array(list(angle_deltas.values()))))
            logging.debug('Max position delta: {}'.format(max_position_delta))
            logging.debug('Max angle delta: {}'.format(max_angle_delta))
            reached_steady_state = (max_position_delta < position_delta_threshold
                                    and max_angle_delta < angle_delta_threshold)
            if (reached_steady_state and not self.linear_velocity_clamps
                    and not self.angular_velocity_clamps):
                break
        return (initial_step - self.step, bounds_removed_items)

    def _add_gripper(self):
        """Add a gripper. Currently broken and unused."""
        logging.info('Adding gripper')
        file_path = os.path.join(GRIPPER_DIR, GRIPPER, 'base.obj')
        cube_start_orientation = pb.getQuaternionFromEuler([0, 0, 0])
        self.gripper_id = pb.loadURDF(file_path, [0, 0, 3], cube_start_orientation,
                                      physicsClientId=self.client_id)
        logging.info('Gripper assigned id {}'.format(self.gripper_id))
        pos, ori = self.getPose(self.gripper_id, physicsClientId=self.client_id)
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
        visual_shape_id = -1
        if self.client_mode == pb.GUI:
            visual_shape_id = pb.createVisualShape(
                pb.GEOM_BOX, halfExtents=[x_length, y_length, z_length],
                visualFramePosition=[x_center, y_center, z_center],
                rgbaColor=[1, 1, 1, 0.5],
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
        visual_shape_id = -1
        if self.client_mode == pb.GUI:
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
        logging.info('Creating crate.')
        self._add_crate_floor()
        for side in range(4):
            self._add_crate_wall(side)


class ScenePopulator(object):
    """Stochastically populates a Scene with items."""

    def __init__(
            self, scene, min_items=10, max_items=20,
            mean_position=np.array([0, 0, 0.25]), position_ranges=np.array([0.05, 0.05, 0.05]),
            min_lateral_friction=0.4, max_lateral_friction=0.6,
            min_spinning_friction=0.0, max_spinning_friction=0.01,
            min_rolling_friction=0.0, max_rolling_friction=0.01,
            sim_height_timeout=1000, sim_height_threshold=0.15, sim_check_interval=100,
            sim_position_delta_threshold=0.004, sim_angle_delta_threshold=np.pi / 32,
    ):
        """Load the database of item models from the filesystem."""
        self.scene = scene
        self.sim_height_timeout = sim_height_timeout
        self.sim_height_threshold = sim_height_threshold
        self.sim_check_interval = sim_check_interval
        self.sim_position_delta_threshold = sim_position_delta_threshold
        self.sim_angle_delta_threshold = sim_angle_delta_threshold

        # Random distribution parameters
        self.min_items = min_items
        self.max_items = max_items
        self.mean_position = mean_position
        self.position_ranges = position_ranges
        self.min_lateral_friction = min_lateral_friction
        self.max_lateral_friction = max_lateral_friction
        self.min_spinning_friction = min_spinning_friction
        self.max_spinning_friction = max_spinning_friction
        self.min_rolling_friction = min_rolling_friction
        self.max_rolling_friction = max_rolling_friction

        # Item loading
        self.full_item_database = sorted(os.listdir(URDF_DIR))
        self.excluded_items = {
            # Invalid objects
            '31340e691294a750d30ee0b5a9888c0b', '38dd2a8d2c984e2b6c1cd53dbc9f7b8e',
            '3c80c41399d2c92334fb047a3245866d', '3f497f8d7dd8922a57e59dddfaae7826',
            '4fcd289d3e82bb08588f04a271bfa5eb', '68582543c4c6d0bccfdfe3f21f42a111',
            '9a52843cc89cd208362be90aaa182ec6', 'a4584986b4baf1479a2840934f7f84bc',
            'a86d587f38569fdf394a7890920ef7fd', 'bacef3777b91b02f29c5890b07f3a65',
            'c09e3db27668639a69fba573ec0b31f3', 'c453274b341f8c4ec2b9bcaf66ea9919',
            'dc0c4db824981b8cf29c5890b07f3a65', 'pliers_standard',
            # Oversized objects
            'f44301a26ca7f57c70d5704d62bc4186', 'a1106fcebabd81743dbb1d116dac64cc',
            '8d3290b8f19761f4ea1322d8b306fad1', '9a65ee8f544ce558965094dba2eb878a',
            '25f6f2002f9287ff6c142f9dd7073357',
            # Excessively complex objects
            '4fcd289d3e82bb08588f04a271bfa5eb1', '5d803107b8a9aec8724d867867ccd9fb',
            '5e42608cac0cb5e94962fcaf2d60c3de', '20a5672ab6767a436fdde33224151fa5',
            '6930c4d2e7e880b2e20e92c5b8147e4a', 'b6f30c63c946c286cf6897d8875cfd5e',
            'b46ad21b7126049842ca7cc070f21ed3', 'ca95ce22929fd53a570c6c691c987a8',
            'd85daabdcd4481becdd3fedfdba1457f', 'df1d8567cf192c61b2cc746911f13ff9',
            #bottle
            '5ad47181a9026fc728cc22dce7529b69', 
            #large coke bottle
            '6b810dbc89542fd8a531220b48579115' ,
            #giant bottle that doesn't fit
            '799397068de1ae1c4587d6a85176d7a0',
            #another bottle that deosn't fin
            'fa44223c6f785c60e71da2487cb2ee5b'

            #For reference
            #473758ca6cb0506ee7697d561711bd2b -> banana
            #e6f95cfb9825c0f65070edcf21eb751c -> round water bottle

        }
        logging.debug('Excluding {} items'.format(len(self.excluded_items)))
        self.item_database = sorted(list(set(
            self.full_item_database) - self.excluded_items
        ))
        logging.debug('Available items: {}'.format(self.item_database))

    def _sample_num_items(self):
        """Sample a uniform random distribution for the number of items to add."""
        return random.randint(self.min_items, self.max_items)

    def _sample_item(self, with_replacement=False):
        """Sample a uniform random distribution for the item to add."""
        if with_replacement:
            available_items = self.item_database
        else:
            available_items = list(set(self.item_database) - set(self.scene.item_names.values()))
        sampled = random.choice(available_items)
        return sampled

    def _sample_position(self):
        """Sample a uniform random distribution for the position of the item to add."""
        return np.random.uniform(self.mean_position - self.position_ranges / 2,
                                 self.mean_position + self.position_ranges / 2)

    def _sample_orientation(self):
        """Sample a uniform random distribution for the orientation of the item to add."""
        euler = [random.uniform(0, 2 * math.pi) for _ in range(3)]
        return pb.getQuaternionFromEuler(euler)

    def _sample_lateral_friction(self):
        """Sample a uniform random distribution for the lateral friction of the item to add."""
        return random.uniform(self.min_lateral_friction, self.max_lateral_friction)

    def _sample_spinning_friction(self):
        """Sample a uniform random distribution for the spinning friction of the item to add."""
        return random.uniform(self.min_spinning_friction, self.max_spinning_friction)

    def _sample_rolling_friction(self):
        """Sample a uniform random distribution for the rolling friction of the item to add."""
        return random.uniform(self.min_rolling_friction, self.max_rolling_friction)

    def add_item(self, unique_items=True):
        """Add a random item to the scene."""
        return self.scene.add_item(
            self._sample_item(with_replacement=(not unique_items)),
            self._sample_position(), self._sample_orientation(),
            lateral_friction=self._sample_lateral_friction(),
            spinning_friction=self._sample_spinning_friction(),
            rolling_friction=self._sample_rolling_friction()
        )

    def simulate_to_height(self, item_id, height_threshold, check_interval=1000):
        """Simulate the scene until the specified item has fallen below the specified height."""
        initial_step = self.scene.step
        while self.scene.step - initial_step < self.sim_height_timeout:
            self.scene.simulate(steps=check_interval)
            pose = self.scene.get_pose(item_id)
            if pose[0][2] < height_threshold:
                return self.scene.step - initial_step
        logging.warn('Timed out while waiting for item {} to fall into the crate.'
                     .format(item_id))

    def add_items(self, num_items=None):
        """Stochastically populate a scene by randomly dropping items into a crate."""
        if num_items is None:
            num_items = self._sample_num_items()
        logging.info('Adding {} items.'.format(num_items))
        for i in range(num_items):
            (item_id, _, _) = self.add_item()
            self.simulate_to_height(
                item_id, self.sim_height_threshold, self.sim_check_interval
            )
            removed_items = self.scene.remove_bounds_items()
            if removed_items:
                logging.info('Removed items out of simulation bounds: {}'
                             .format(removed_items))
        logging.info('Simulating to steady state.')
        self.scene.simulate_to_steady_state(
            position_delta_threshold=self.sim_position_delta_threshold,
            angle_delta_threshold=self.sim_angle_delta_threshold
        )
        logging.info('Reached steady state.')


def main():
    """Initialize and populate a random scene and simulate it."""
    scene = Scene(show_gui=True)

    populator = ScenePopulator(scene)
    populator.add_items(num_items=10)
    logging.info('Finished initializing scene.')

    print(scene.get_item_poses(to_euler=True))

    input('Press any key to clear the scene...')
    scene.remove_all_items()

    input('Press any key to populate the scene again...')
    populator.add_items()
    logging.info('Finished initializing scene.')

    input('Press any key to end...')


if __name__ == '__main__':
    main()
