"""Creates a bandit policy by picking the highest probability grasp.

Calculates final reward over epochs.
"""

import logging
import math

from crate import CrateMDP, random_baseline, highest_first_baseline, lowest_first_baseline, Action

from scene import Scene, ScenePopulator


SAVE_FILE = "results_lowest.txt"


def main():
    """Run a bandit policy."""
    scene = Scene(show_gui=True)
    scene_populator = ScenePopulator(scene)
    env = CrateMDP(scene, scene_populator)
    discount = 0.9
    num_episodes = 100
    rewards_for_all_episodes = 0
    f = open (SAVE_FILE, "w")
    f.write ("episode reward avg\n")
    for episode in range(num_episodes):
        state = env.reset()
        logging.info('Starting episode {}'.format(episode))
        discounted_return = 0
        step = 0

        actions = env.get_actions(state)   # calls collision checker already
        print ('episode number')
        print episode
        print ('Number of avail actions')
        print len(actions)

        while True:
            # find action with highest metric

            #action = random_baseline(state)
            action = lowest_first_baseline(state)
            #action = highest_first_baseline(state)
  
            #DEX NET ACTION
            #actions = env.get_actions(state)   # calls collision checker already
            #action = actions[0]

            # best_metric = actions[0].metric
            # for act in actions:
            #     metric = act.metric
            #     if metric > best_metric:
            #         action = act
            #         best_metric = metric
            # print(action.item_id)
            logging.info('Attempting to remove item {}...'.format(action.item_id))

            #NOTE: turn check_next = False to not check if there are valid actions in the next state, this speeds things up a bit
            (state, reward, done) = env.step(action, check_next = False)
            logging.info('Received reward {}'.format(reward))
            print "Done status: "
            print done
            discounted_return += math.pow(discount, step) * reward
            step += 1
            if done:
                break
        rewards_for_all_episodes += discounted_return
        print ('Episode {} accumulated a discounted return of {}'
                     .format(episode, discounted_return))

        avg = rewards_for_all_episodes / (episode + 1)
        f.write (str(episode + 1) + " " + str(discounted_return) + " " + str(avg) + "\n")
        f.flush ()

        print 'Average reward per episode: ' + str()

        logging.info('Episode {} accumulated a discounted return of {} and running average of {}'
                     .format(episode, discounted_return, avg))


    f.close ()

    try:
        input('Press any key to end...')
    except Exception as e:
        pass




if __name__ == '__main__':
    main()
