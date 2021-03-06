"""Crate/bin-picking environments for running policies."""
import logging
import math
import random
import pdb
import utils.general as general
import autolab_core
from action_finder import ActionFinder


import numpy as np

from scene import Scene, ScenePopulator
from action_finder import Action, ActionFinder

NUM_ITEMS = 10
PENALTY_FOR_COLIFT = -10 #penalty for co-lifting other objectsre
CONSECUTIVE_FAILURE_DONE_THRESHOLD = 10

ACTION_DIMS = 5
ACTION_CHOICES_MAX = 10


class CrateMDP(object):
    """An environment for the crate/bin-picking task with OpenAI gym-esque interface."""

    def __init__(self, scene, scene_populator, pomdp=False, sim_remove_velocity=[0, 0, 5],
                 sim_position_delta_threshold=0.004, sim_angle_delta_threshold=np.pi / 32,
                 ignore_feasibility=False):
        """Store the Scene and ScenePopulator to use for managing the environment."""
        self.scene = scene
        self.scene_populator = scene_populator
        self.pomdp = pomdp
        self.sim_remove_velocity = sim_remove_velocity
        self.sim_position_delta_threshold = sim_position_delta_threshold
        self.sim_angle_delta_threshold = sim_angle_delta_threshold
        self.encoded_observation_shape = [self.scene_populator.max_items,
                                          len (self.scene_populator.item_database) + 7]
        self._current_candidate_actions = None
        self.af = ActionFinder()
        self.ignore_feasibility = ignore_feasibility
        self.consecutive_failures = 0

    def encode_state(self, state):
        one_hot_item_ids = np.zeros([self.scene_populator.max_items, 
                                    len(self.scene_populator.item_database)])
        for (i, item_name) in enumerate (state['item_ids']):
            item_idx = self.scene_populator.item_database.index (item_name)
            one_hot_item_ids[i, item_idx] = 1
        poses = np.zeros((self.scene_populator.max_items, 7))  # 7 = position (3) + orientation (4)
        for i, item_name in enumerate (state['item_ids']):

            item_id = state["item_ids"][item_name] 
            poses[i, :3] = state['poses'][item_id][0]
            poses[i, 3:] = state['poses'][item_id][1]

        return np.ndarray.flatten(np.hstack([poses, one_hot_item_ids]))

    def encode_action (self, action, state):
        g = action.grasp
        item_id = action.item_id
        obj_pose = state['poses'][item_id]
        g_to_w = general.grasp_to_world(obj_pose, g.T_grasp_obj)
        axis = g_to_w.x_axis
        angle = general.angle_between(np.array([1, 0, 0]), axis)
        grasp_loc_world = g_to_w.inverse().translation
        return [grasp_loc_world[0], grasp_loc_world[1], grasp_loc_world[2], angle, action.metric] 

    def encode_action_choices (self, action_choices, state):
        #encode action_choices into x,y,d,theta
        #returns action_choices and mask
        count = len (action_choices)
        assert count <= self.get_action_choices_max ()
        encoded = np.zeros ([self.get_action_choices_max (), self.get_action_dims ()])
        mask = np.zeros (self.get_action_choices_max (), dtype=bool)
        for idx, a in enumerate (action_choices):
            encoded[idx] = self.encode_action(a, state)
            mask [idx] = True

        #print "encoding actions"
        #print encoded
        #print mask
        return encoded, mask


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

        return action.metric

    def _remove_item(self, item_id, pull_remove=True):
        """Remove an item and simulate until the system reaches steady state again."""
        if pull_remove:
            self.scene.clamp_item_velocity(item_id, linear=self.sim_remove_velocity,
                                           angular=[0, 0, 0])
        else:
            self.scene.remove_item(item_id)
        (_, bounds_removed_items, stuck) = self.scene.simulate_to_steady_state(
            position_delta_threshold=self.sim_position_delta_threshold,
            angle_delta_threshold=self.sim_angle_delta_threshold)

        if stuck:
            try:
                self.scene.remove_item (item_id)
                (_, bounds_removed_items, stuck_again) = self.scene.simulate_to_steady_state(
                position_delta_threshold=self.sim_position_delta_threshold,
                angle_delta_threshold=self.sim_angle_delta_threshold)
                if stuck_again:
                    print "WARNING: static removal was also stuck"
            except KeyError:
                pass

        try:
            bounds_removed_items.remove(item_id)
        except KeyError:
            pass
        return bounds_removed_items

    def get_action_dims (self):
        return ACTION_DIMS

    def get_action_choices_max (self):
        return ACTION_CHOICES_MAX

    def get_encoded_observation_shape (self):
        return self.encoded_observation_shape

    def reset(self):
        """Reset environment to state sampled from distribution of initial states."""
        self.scene.remove_all_items()
        self.scene_populator.add_items(num_items=NUM_ITEMS)
        observation = self._observe_current()
        self._current_candidate_actions = (None if self.ignore_feasibility
                                           else self.af.find_alt(observation))
        self.consecutive_failures = 0
        return observation

    def step(self, action):
        """Take an action on the environment."""
        success_probability = self._get_success_probability(action)
        success = np.random.binomial(1, success_probability)
        reward = success
        if success:
            bounds_removed_items = self._remove_item(action.item_id)
            reward += PENALTY_FOR_COLIFT * len(bounds_removed_items)  # Penalize for knocking other items out
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
        observation = self._observe_current()   # may need to change for POMDP
        done = False
        if len(self.scene.item_ids) == 0:
            done = True
            logging.info('Environment terminated: no remaining items.')
        if not self.ignore_feasibility:
            self._current_candidate_actions = self.af.find_alt(observation)
            if len(self._current_candidate_actions) == 0:
                logging.info('Environment terminated: no remaining candidate actions.')
                done = True
        if self.consecutive_failures > CONSECUTIVE_FAILURE_DONE_THRESHOLD:
            logging.info('Environment terminated: too many consecutive failed grasp attempts.')
            done = True

        return (observation, reward, done)

    def get_current_candidate_actions(self):
        return self._current_candidate_actions

