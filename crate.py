"""Crate/bin-picking environments for running policies."""

import collections
import logging
import math
import random

import autolab_core

import dexnet

import numpy as np

from scene import Scene, ScenePopulator

# Import input/raw-input with python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass


Action = collections.namedtuple('Action', ['item_id', 'gripper_pose'])

NUM_ITEMS = 10
DEX_NET_PATH = '../dex-net/'
DB_NAME = 'dexnet_2.hdf5'
GRIPPER_NAME = 'yumi_metal_spline'
GRIPPER_REL_PATH = 'data/grippers/'
GRASP_METRIC = 'robust_ferrari_canny'


class CrateMDP(object):
    """An environment for the crate/bin-picking task with OpenAI gym-esque interface."""

    def __init__(self, scene, scene_populator, mdp=True, sim_remove_velocity=[0, 0, 2],
                 sim_position_delta_threshold=0.004, sim_angle_delta_threshold=np.pi / 32):
        """Store the Scene and ScenePopulator to use for managing the environment."""
        self.scene = scene
        self.scene_populator = scene_populator
        self.mdp = mdp
        self.sim_remove_velocity = sim_remove_velocity
        self.sim_position_delta_threshold = sim_position_delta_threshold
        self.sim_angle_delta_threshold = sim_angle_delta_threshold
        self.dn = dexnet.DexNet()
        self.dn.open_database(DEX_NET_PATH + DB_NAME, create_db=True)
        self.dn.open_dataset('3dnet')
        self.gripper_name = GRIPPER_NAME
        self.gripper = dexnet.grasping.gripper.RobotGripper.load(
            GRIPPER_NAME, DEX_NET_PATH + GRIPPER_REL_PATH
        )
        self.gripper_pose = ([0, 0, 1], [1, 0, 0, 0])
        self.cc_approach_dist = 1.0
        self.cc_delta_approach = 0.1    # may need tuning

    def _get_current_state(self):
        return {
            'poses': self.scene.get_item_poses(),
            'item_ids': self.scene.item_ids,
            'item_names': self.scene.item_names,
        }

    def _observe_current(self):
        """Generate observation of environment, namely the current state.

        Override this to implement an alternative observation model for a POMDP.
        """
        return self._get_current_state()

    def _get_success_probability(self, action):
        """Compute the probability of success of an action."""
        # call dex-net to compute probability of grasp success.
        # action has to be object ID and gripper pose (gripper center position as x,y,z and
        # gripper orientation as quaternion) in global frame.
        # TODO: implement this.
        # TODO: how does this work? because the current ferrari canny values are too small
        return 0.8

    def _remove_item(self, item_id, pull_remove=True):
        """Remove an item and simulate until the system reaches steady state again."""
        if pull_remove:
            self.scene.clamp_item_velocity(item_id, linear=self.sim_remove_velocity,
                                           angular=[0, 0, 0])
        else:
            self.scene.remove_item(item_id)
        (_, bounds_removed_items) = self.scene.simulate_to_steady_state(
            position_delta_threshold=self.sim_position_delta_threshold,
            angle_delta_threshold=self.sim_angle_delta_threshold
        )
        try:
            bounds_removed_items.remove(item_id)
        except KeyError:
            pass
        return bounds_removed_items

    def reset(self):
        """Reset environment to state sampled from distribution of initial states."""
        self.scene.remove_all_items()
        self.scene_populator.add_items(num_items=NUM_ITEMS)
        return self._observe_current()

    def step(self, action):
        """Take an action on the environment."""
        success_probability = self._get_success_probability(action)
        success = np.random.binomial(1, success_probability)
        reward = success
        if success:
            bounds_removed_items = self._remove_item(action["item_id"])
            reward -= len(bounds_removed_items)  # Penalize for knocking other items out
        observation = self._observe_current()   # may need to change for POMDP
        done = (len(self.scene.item_ids) == 0)
        actions = self.get_actions(observation) # make sure there are still further actions to be executed
        if len(actions) == 0:
            reward -= 10000
            done = True
        return (observation, reward, done)

    def check_collisions(self, state, actions):
        """Filter the provided actions for actions which don't cause collisions."""
        # TODO: implement this.
        if not self.mdp:
            return []

        gcc = dexnet.grasping.GraspCollisionChecker(self.gripper)

        # Add all objects to the world frame
        item_names = state['item_names']
        poses = state['poses']
        graspables = {}
        for item_id, pose in poses.items():
            # TODO: need to figure out how rigidtransform works
            graspable = self.dn.dataset.graspable(item_names[item_id])
            gcc.add_graspable_object(graspable)
            graspables[item_id] = graspable

        for idx in reversed(range(len(actions))):
            action = actions[idx]
            # TODO: rigidtransform
            pose = poses[action['item_id']]
            rot_obj = autolab_core.RigidTransform.rotation_from_quaternion(pose[1])
            rot_grip = autolab_core.RigidTransform.rotation_from_quaternion(self.gripper_pose[1])
            world_to_obj = autolab_core.RigidTransform(rot_obj, pose[0], 'world', 'obj')
            world_to_grip = autolab_core.RigidTransform(rot_grip, self.gripper_pose[0],
                                                        'world', 'gripper')
            obj_to_world = world_to_obj.inverse()
            obj_to_grip = world_to_grip.dot(obj_to_world)
            # now have RigidTransform of gripper wrt object, so can pass to collision check

            # homogT_obj_wrt_world = np.vstack((
            #     np.vstack((rotMat.T, pose[0])).T,
            #     np.array([0,0,0,1])
            # ))
            # print homogT_obj_wrt_world
            # homogT_world_wrt_obj = np.linalg.inv(homogT_obj_wrt_world)
            # homogT_gripper_wrt_obj = homogT_world_wrt_obj
            gcc.set_graspable_object(graspables[action['item_id']])
            # TODO: fix this code so it works
            # if gcc.grasp_in_collision () or gcc.collides_along_approach ():
            #     del actions[idx]
            grasp_collide = gcc.grasp_in_collision(obj_to_grip.inverse(), action['name'])
            # approach_dist and delta_approach params to play with
            approach_collide = gcc.collides_along_approach(
                action['grasp'], self.cc_approach_dist, self.cc_delta_approach,
                action['name']
            )
            if grasp_collide or approach_collide:
                del actions[idx]

        return actions

    def get_actions(self, state):
        """Get all actions given the current state.

        This can only be used in mdp mode.
        """
        if not self.mdp:
            return []
        item_names = state['item_names']
        poses = state['poses']
        actions = []

        for item_id, pose in poses.items():
            name = item_names[item_id]
            grasps, metrics = self.dn.get_grasps(name, GRIPPER_NAME, GRASP_METRIC)
            actions.append({
                'item_id': item_id,
                'name': name,
                'grasp': grasps[0],
                'metric': metrics[0]
            })
        print "pre-pruned actions"
        print actions
        actions = self.check_collisions(state, actions)
        return actions


def random_baseline(state):
    return Action(random.choice(list(state.keys())), None)

def highest_first_baseline(state):
    item_heights = {item: pose[0][2] for (item, pose) in state['poses'].items()}
    return Action(max(item_heights, key=item_heights.get), None)

def lowest_first_baseline(state):
    item_heights = {item: pose[0][2] for (item, pose) in state['poses'].items()}
    return Action(min(item_heights, key=item_heights.get), None)


def main():
    """Run a baseline heuristic policy on the CrateMDP environment."""
    scene = Scene(show_gui=True)
    scene_populator = ScenePopulator(scene)
    env = CrateMDP(scene, scene_populator)
    discount = 0.9
    num_episodes = 5
    average_discounted_return = 0
    for episode in range(num_episodes):
        state = env.reset()
        logging.info('Starting episode {}'.format(episode))
        discounted_return = 0
        step = 0
        while True:
            # action = random_baseline(state)
            action = lowest_first_baseline(state)
            # action = highest_first_baseline(state)
            logging.info('Attempting to remove item {}...'.format(action["item_id"]))
            (state, reward, done) = env.step(action)
            logging.info('Received reward {}'.format(reward))
            discounted_return += math.pow(discount, step) * reward
            step += 1
            if done:
                break
        logging.info('Episode {} accumulated a discounted return of {}'
                     .format(episode, discounted_return))
        average_discounted_return += discounted_return / num_episodes
    logging.info('Average discounted return over {} episodes is {}'
                 .format(num_episodes, average_discounted_return))
    input('Press any key to end...')


if __name__ == '__main__':
    main()
