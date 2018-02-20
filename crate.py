"""Crate/bin-picking environments for running policies."""

import collections
import logging
import math
import random

import numpy as np

from scene import Scene, ScenePopulator

Action = collections.namedtuple('Action', ['item_id', 'gripper_pose'])


class CrateMDP(object):
    """An environment for the crate/bin-picking task with OpenAI gym-esque interface."""

    def __init__(self, scene, scene_populator, sim_position_delta_threshold=0.002,
                 sim_angle_delta_threshold=np.pi / 64):
        """Store the Scene and ScenePopulator to use for managing the environment."""
        self.scene = scene
        self.scene_populator = scene_populator
        self.sim_position_delta_threshold = sim_position_delta_threshold
        self.sim_angle_delta_threshold = sim_angle_delta_threshold

    def _get_current_state(self):
        return self.scene.get_item_poses(to_euler=True)

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

    def reset(self):
        """Reset environment to state sampled from distribution of initial states."""
        self.scene.remove_all_items()
        self.scene_populator.add_items()
        return self._observe_current()

    def step(self, action):
        """Take an action on the environment."""
        success_probability = self._get_success_probability(action)
        success = np.random.binomial(1, success_probability)
        reward = success
        if success:
            self.scene.remove_item(action.item_id)
            (_, freefall_removed_items) = self.scene.simulate_to_steady_state(
                position_delta_threshold=self.sim_position_delta_threshold,
                angle_delta_threshold=self.sim_angle_delta_threshold
            )
            reward -= len(freefall_removed_items)  # Penalize for knocking other items out
        observation = self._observe_current()
        done = (len(self.scene.item_ids) == 0)
        return (observation, reward, done)

    def check_collisions(self, actions):
        """Filter the provided actions for actions which don't cause collisions."""
        # TODO: implement this.
        return actions


def main():
    """Run a baseline random policy on the CrateMDP environment."""
    scene = Scene()
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
            action = Action(random.choice(list(state.keys())), None)
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
