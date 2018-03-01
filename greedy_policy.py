"""Creates a bandit policy by picking the highest probability grasp.

Calculates final reward over epochs.
"""

import logging
import math

from crate import CrateMDP

from scene import Scene, ScenePopulator


def main():
    """Run a bandit policy."""
    scene = Scene(show_gui=True)
    scene_populator = ScenePopulator(scene)
    env = CrateMDP(scene, scene_populator)
    discount = 1
    num_episodes = 1
    average_discounted_return = 0
    for episode in range(num_episodes):
        state = env.reset()
        logging.info('Starting episode {}'.format(episode))
        discounted_return = 0
        step = 0

        actions = env.get_actions(state)   # calls collision checker already
        print('ACTIONS')
        print(actions)
        print('STATE')
        print(state)
        while True:
            # find action with highest metric
            actions = env.get_actions(state)   # calls collision checker already
            action = actions[0]
            best_metric = actions[0].metric
            for act in actions:
                metric = act.metric
                if metric > best_metric:
                    action = act
                    best_metric = metric
            print(action.item_id)
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

    try:
        input('Press any key to end...')
    except Exception as e:
        pass


if __name__ == '__main__':
    main()
