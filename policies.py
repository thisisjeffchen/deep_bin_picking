"""Support for implementation of policies, and baseline policies."""
import logging
import math
import random
import pdb

from env.scene import Scene, ScenePopulator
from env.crate import Action, CrateMDP
from env.action_finder import Action, ActionFinder

class Policy(object):
    def choose_action(self, state):
        pass

class PolicyRunner(object):
    def __init__(self, scene, scene_populator, env, policy, discount=0.9):
        self.scene = scene
        self.scene_populator = scene_populator
        self.env = env
        self.policy = policy
        self.discount = discount

    def start_episode(self):
        self.state = self.env.reset()
        self.discounted_return = 0
        self.step = 0

    def step_episode(self):
        action = self.policy.choose_action(self.state)
        if action == None:
            return False
        logging.info('Attempting to remove item {}...'.format(action.item_id))
        (state, reward, done) = self.env.step(action)
        logging.info('Received reward {}'.format(reward))
        self.state = state
        self.discounted_return += math.pow(self.discount, self.step) * reward
        self.step += 1
        return not done

    def run_episode(self):
        self.start_episode()
        while self.step_episode():
            pass
        return self.discounted_return

class ReturnsLogger(object):
    def __init__(self, filepath=None, description=None):
        self.filepath = filepath
        self.description = description
        self.file = None

    def __enter__(self):
        if self.filepath is None:
            return
        self.file = open(self.filepath, 'w')
        self.file.write('Discounted returns in episodes\n')
        if self.description is not None:
            self.file.write('{}\n'.format(description))

    def __exit__(self, *args):
        if self.file is not None:
            self.file.close()

    def log_returns(self, episode, discounted_return, average_discounted_returns):
        logging.info('Episode {} accumulated a discounted return of {}'.format(
            episode, discounted_return
        ))
        logging.info('Average reward per episode over past {} episodes: {}'.format(
            episode + 1, average_discounted_returns
        ))
        if self.returns_file is not None:
            self.returns_file.write('{} {} {}\n'.format(
                episode, discounted_return, average_discounted_returns
            ))
            self.returns_file.flush()

class PolicyTester(object):
    def __init__(self, policy_runner, returns_logger, num_episodes=500):
        self.policy_runner = policy_runner
        self.num_episodes = num_episodes
        self.episode = 0
        self.total_discounted_returns = 0
        self.average_discounted_returns = 0
        self.returns_logger = returns_logger

    def run_episode(self):
        logging.info('Running episode {}...'.format(self.episode))
        discounted_return = self.policy_runner.run_episode()
        self.total_discounted_returns += discounted_return
        average_discounted_returns = self.total_discounted_returns / (episode + 1)
        self.returns_logger.log_returns(
            episode + 1, discounted_return, average_discounted_returns
        )
        self.episode += 1

    def run_episodes(self):
        logging.info('Running {} episodes...'.format(self.num_episodes))
        while (self.episode < self.num_episodes):
            self.run_episode()

class HighestFirstBaseline(Policy):
    def choose_action(self, state):
        item_heights = {item: pose[0][2] for (item, pose) in state['poses'].items()}
        if len(item_heights) == 0:
            return None
        item_id = max(item_heights, key=item_heights.get)
        return Action(item_id, state['item_names'][item_id], None, 1.0)

def LowestFirstBaseline(Policy):
    def choose_action(self, state):
        item_heights = {item: pose[0][2] for (item, pose) in state['poses'].items()}
        if len(item_heights) == 0:
            return None
        item_id = min(item_heights, key=item_heights.get)
        return Action(item_id, state['item_names'][item_id], None, 1.0)

class RandomBaseline(Policy):
    def choose_action(self, state):
        # TODO: only choose from available actions
        (item_id, item_name) = random.choice(list(state['poses'].items()))
        return Action(item_id, item_name, None, None)

class GreedyBaseline(Policy):
    def choose_action(self, state):
        # TODO: implement a greedy baseline
        (item_id, item_name) = random.choice(list(state['poses'].items()))
        return Action(item_id, item_name, None, None)

def main():
    """Run a baseline heuristic policy on the CrateMDP environment."""
    scene = Scene(show_gui=False)
    scene_populator = ScenePopulator(scene)
    env = CrateMDP(scene, scene_populator)
    policy = HighestFirstBaseline()
    policy_runner = PolicyRunner(scene, scene_populator, env, policy)
    with ReturnsLogger('policies_test') as returns_logger:
        policy_tester = PolicyTester(policy_runner, returns_logger)
        policy_tester.run_episodes()


if __name__ == '__main__':
    main()
