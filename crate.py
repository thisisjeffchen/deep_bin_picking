"""Crate/bin-picking environments for running policies."""

import collections
import logging
import math
import random
import dexnet

import numpy as np

from scene import Scene, ScenePopulator

Action = collections.namedtuple('Action', ['item_id', 'gripper_pose'])

NUM_ITEMS = 1
DEX_NET_PATH = "../dex-net/"
DB_NAME = "dexnet_2.hdf5"
GRIPPER_NAME = "yumi_metal_spline"
GRIPPER_REL_PATH = "data/grippers/"
GRASP_METRIC = "robust_ferrari_canny"

class CrateMDP(object):
    """An environment for the crate/bin-picking task with OpenAI gym-esque interface."""

    def __init__(self, scene, scene_populator, mdp = True, sim_remove_velocity=[0, 0, 2],
                 sim_position_delta_threshold=0.004, sim_angle_delta_threshold=np.pi / 32):
        """Store the Scene and ScenePopulator to use for managing the environment."""
        self.scene = scene
        self.scene_populator = scene_populator
        self.mdp = mdp
        self.sim_remove_velocity = sim_remove_velocity
        self.sim_position_delta_threshold = sim_position_delta_threshold
        self.sim_angle_delta_threshold = sim_angle_delta_threshold
        self.dn = dexnet.DexNet ()
        self.dn.open_database (DEX_NET_PATH + DB_NAME, create_db = True) 
        self.dn.open_dataset ('3dnet')
        self.gripper_name = GRIPPER_NAME

    def _get_current_state(self):
        return {"poses": self.scene.get_item_poses(to_euler=True),
                "dex": self.scene.item_ids }

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
        self.scene_populator.add_items(num_items = NUM_ITEMS)
        return self._observe_current()

    def step(self, action):
        """Take an action on the environment."""
        success_probability = self._get_success_probability(action)
        success = np.random.binomial(1, success_probability)
        reward = success
        if success:
            bounds_removed_items = self._remove_item(action.item_id)
            reward -= len(bounds_removed_items)  # Penalize for knocking other items out
        observation = self._observe_current()
        done = (len(self.scene.item_ids) == 0)
        return (observation, reward, done)

    def check_collisions(self, actions):
        """Filter the provided actions for actions which don't cause collisions."""
        # TODO: implement this.
        if not self.mdp:
            return []
        #graspable = dn.dataset.graspable (dex_id)   
        return actions
    


    def get_actions(self, state):
        '''Gets all actions given the current state, this can only be used
           in mdp mode ''' 
        if not self.mdp:
            return []
        dex_keys = state["dex"]
        poses = state["poses"]
        actions = []

        for item_id, pose in poses.items ():
            dex_id = dex_keys[item_id]
            grasps, metrics = self.dn.get_grasps (dex_id, GRIPPER_NAME, GRASP_METRIC)
            actions.append ({"item_id": item_id,
                             "dex_id": dex_id,
                             "grasp" : grasps[0],
                             "metric": metrics[0]})                                        
        actions = self.check_collisions (actions)
        return actions


def random_baseline(state):
    return Action(random.choice(list(state.keys())), None)

def highest_first_baseline(state):
    item_heights = {item: pose[0][2] for (item, pose) in state.items()}
    return Action(max(item_heights, key=item_heights.get), None)

def lowest_first_baseline(state):
    item_heights = {item: pose[0][2] for (item, pose) in state.items()}
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
            logging.info('Attempting to remove item {}...'.format(action.item_id))
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
