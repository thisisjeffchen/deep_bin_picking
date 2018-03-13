"""Support for implementation of policies, and baseline policies."""
import logging
import math
import random
import pdb
from env.crate import Action

from scene import Scene, ScenePopulator
from action_finder import Action, ActionFinder

# Import input/raw-input with python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

class Policy(object):
    def choose_action(self, state):
        pass

class HighestFirstBaseline(Policy):
    def choose_action(self, state):
        item_heights = {item: pose[0][2] for (item, pose) in state['poses'].items()}
        if (len(item_heights) == 0):
       	    return None
        item_id = max(item_heights, key=item_heights.get)
        return Action(item_id, state['item_names'][item_id], None, 1.0)

def LowestFirstBaseline(Policy):
    def choose_action(self, state):
        item_heights = {item: pose[0][2] for (item, pose) in state['poses'].items()}
        if (len(item_heights) == 0):
            return None
        item_id = min(item_heights, key=item_heights.get)
        return Action(item_id, state['item_names'][item_id], None, 1.0)

class RandomBaseline(Policy):
    def choose_action(self, state):
        (item_id, item_name) = random.choice(list(state['poses'].items()))
        return Action(item_id, item_name, None, None)

def main():
    """Run a baseline heuristic policy on the CrateMDP environment."""
    scene = Scene(show_gui=True)
    scene_populator = ScenePopulator(scene)
    env = CrateMDP(scene, scene_populator)
    policy = HighestFirstBaseline()
    discount = 0.9
    num_episodes = 5
    average_discounted_return = 0
    for episode in range(num_episodes):
        state = env.reset()
        logging.info('Starting episode {}'.format(episode))
        discounted_return = 0
        step = 0
        while True:
            action = policy(state)
            if action == None:
                done = True
                break
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
