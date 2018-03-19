"""Support for implementation of policies, and baseline policies."""
import contextlib
import logging
import os
import math
import operator
import random
import argparse
import pdb

from env.scene import Scene, ScenePopulator
from env.crate import Action, CrateMDP
from env.action_finder import Action, ActionFinder

class flag(object):
    def __init__(self, ts):
        self.tiny_space = ts

class Policy(object):
    def __init__(self, env):
        self.env = env

    def choose_current_action(self):
        raise NotImplementedError

    def choose_action(self, state):
        raise NotImplementedError

class InfeasiblePolicy(Policy):
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
        try:
            action = self.policy.choose_current_action()
        except NotImplementedError:
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
        return (self.discounted_return, self.step)

@contextlib.contextmanager
def returns_log(filepath=None, description=None):
    file = None
    if filepath is not None:
        file = open(filepath, 'w')
        file.write('episode,discountedReturn,averageDiscountedReturn,episodeLength,averageEpisodeLength\n')
        if description is not None:
            file.write('{}\n'.format(description))
    yield file
    if file is not None:
        file.close()


class PolicyTester(object):
    def __init__(self, policy_runner, returns_logfile, num_episodes=500):
        self.policy_runner = policy_runner
        self.num_episodes = num_episodes
        self.episode = 0
        self.total_discounted_returns = 0
        self.average_discounted_returns = 0
        self.total_timesteps = 0
        self.average_episode_length = 0
        self.returns_logfile = returns_logfile

    def run_episode(self):
        logging.info('Running episode {}...'.format(self.episode))
        (discounted_return, episode_length) = self.policy_runner.run_episode()
        self.total_discounted_returns += discounted_return
        self.average_discounted_returns = self.total_discounted_returns / (self.episode + 1)
        self.total_timesteps += episode_length
        self.average_episode_length = self.total_timesteps / (self.episode + 1)
        logging.info('Episode {} accumulated a discounted return of {}'.format(
            self.episode + 1, discounted_return
        ))
        logging.info('Average reward per episode over past {} episodes: {}'.format(
            self.episode + 1, self.average_discounted_returns
        ))
        logging.info('Average episode length over past {} episodes: {}'.format(
            self.episode + 1, self.average_episode_length
        ))
        if self.returns_logfile is not None:
            self.returns_logfile.write('{},{},{},{},{}\n'.format(
                self.episode + 1, discounted_return,
                self.average_discounted_returns, episode_length,
                self.average_episode_length
            ))
            self.returns_logfile.flush()
        self.episode += 1

    def run_episodes(self):
        logging.info('Running {} episodes...'.format(self.num_episodes))
        while (self.episode < self.num_episodes):
            self.run_episode()

class HighestFirstBaseline(InfeasiblePolicy):
    def choose_action(self, state):
        item_heights = {item: pose[0][2] for (item, pose) in state['poses'].items()}
        if len(item_heights) == 0:
            return None
        item_id = max(item_heights, key=item_heights.get)
        return Action(item_id, state['item_names'][item_id], None, 1.0)

class LowestFirstBaseline(InfeasiblePolicy):
    def choose_action(self, state):
        item_heights = {item: pose[0][2] for (item, pose) in state['poses'].items()}
        if len(item_heights) == 0:
            return None
        item_id = min(item_heights, key=item_heights.get)
        return Action(item_id, state['item_names'][item_id], None, 1.0)

class RandomBaseline(Policy):
    def choose_current_action(self):
        actions = self.env.get_current_candidate_actions()
        if not actions:
            return None
        action = random.choice(actions)
        return action

class GreedyBaseline(Policy):
    def choose_current_action(self):
        actions = self.env.get_current_candidate_actions()
        if not actions:
            return None
        action = max(actions, key=operator.attrgetter('metric'))
        return action

class GreedyHighest(Policy):
    def choose_current_action(self):
        actions = self.env.get_current_candidate_actions()
        if not actions:
            return None
        # first action should already be highest, actions are sorted based on 
        action = actions[0]
        return action


BASELINE_POLICIES = {
    'baseline_highest': HighestFirstBaseline,
    'baseline_lowest': LowestFirstBaseline,
    'baseline_random': RandomBaseline,
    'baseline_greedy': GreedyBaseline,
    'greedy_highest': GreedyHighest
}


def test_policy(policy_name, Policy,
                policy_factory_args=[], policy_factory_kwargs={}):
    """Run a policy on the CrateMDP environment."""
    scene = Scene(show_gui=False)
    scene_populator = ScenePopulator(scene)
    f = flag(True)
    env = CrateMDP(scene, scene_populator, flags=f,
                   ignore_feasibility=issubclass(Policy, InfeasiblePolicy))
    policy = Policy(env, *policy_factory_args, **policy_factory_kwargs)
    policy_runner = PolicyRunner(scene, scene_populator, env, policy)
    results_dir = os.path.join('results', policy_name)
    if not os.path.exists(results_dir):
        logging.info('Creating path {}...'.format(results_dir))
        os.makedirs(results_dir)
    with returns_log('{}/results.txt'.format(results_dir, policy_name)) as f:
        policy_tester = PolicyTester(policy_runner, f)
        policy_tester.run_episodes()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'policy', type=str, choices=BASELINE_POLICIES.keys(),
        help='The baseline policy to test.'
    )
    args = parser.parse_args()
    test_policy(args.policy, BASELINE_POLICIES[args.policy])


if __name__ == '__main__':
    main()
