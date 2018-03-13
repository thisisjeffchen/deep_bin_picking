import dexnet
import collections
import autolab_core
import pdb
import random

DEX_NET_PATH = '../dex-net/'
DB_NAME = 'dexnet_2.hdf5'
GRIPPER_NAME = 'yumi_metal_spline'
GRIPPER_REL_PATH = 'data/grippers/'
GRASP_METRIC = 'ferrari_canny'
DEFAULT_NUM_GRASPS_PER_ITEM = 3
GRIPPER_Z_POS = 1
FC_90_THRESHOLD = 0.010809481366594122
MAX_PROB = 0.9
FC_PRUNE_THRESHOLD = 0.001

ACTION_COLLISION_CHECK_MAX_SOFT = 3
ACTION_COLLISION_CHECK_MAX_HARD = 10
ACTION_SKIP_RATE = 11
MIN_RETURN_ACTIONS = 3

Action = collections.namedtuple('Action', ['item_id', 'item_name', 'grasp', 'metric'])

class ActionFinder (object):
    def __init__ (self):
        self.gcc = None
        self.dn = dexnet.DexNet()
        self.dn.open_database(DEX_NET_PATH + DB_NAME, create_db=True)
        self.dn.open_dataset('3dnet')
        self.gripper_name = GRIPPER_NAME
        self.gripper = dexnet.grasping.gripper.RobotGripper.load(
            GRIPPER_NAME, DEX_NET_PATH + GRIPPER_REL_PATH
        )
        self.gripper_rotation =  [1, 0, 0, 0]
        self.cc_approach_dist = 1.0
        self.cc_delta_approach = 0.1    # may need tuning


    def _is_collision_free (self, gcc, item_name, graspable, grasp, pose):
        """
        Makes sure a particular graspable is collision free
        """
        gripper_pose = [pose[0][0], pose[0][1], GRIPPER_Z_POS]

        rot_obj = autolab_core.RigidTransform.rotation_from_quaternion(pose[1])
        rot_grip = autolab_core.RigidTransform.rotation_from_quaternion(self.gripper_rotation)
        world_to_obj = autolab_core.RigidTransform(rot_obj, pose[0], 'world', 'obj')

        world_to_grip = autolab_core.RigidTransform(rot_grip, gripper_pose,
                                                        'world', 'gripper')
        obj_to_world = world_to_obj.inverse()
        obj_to_grip = world_to_grip.dot(obj_to_world)
        # now have RigidTransform of gripper wrt object, so can pass to collision check

        gcc.set_graspable_object(graspable)
        grasp_collide = gcc.grasp_in_collision(obj_to_grip.inverse(), item_name)
        approach_collide = gcc.collides_along_approach(
                grasp, self.cc_approach_dist, self.cc_delta_approach, item_name)

        if grasp_collide or approach_collide:
            #print "collision detected " + item_name
            return False

        return True


    def _create_grasp_collision_checker (self, state):
        gcc = dexnet.grasping.GraspCollisionChecker(self.gripper)

        # Add all objects to the world frame
        item_names = state['item_names']
        poses = state['poses']
        graspables = {}
        for item_id, pose in poses.items():
            graspable = self.dn.dataset.graspable(item_names[item_id])
            rot_obj = autolab_core.RigidTransform.rotation_from_quaternion(pose[1])
            world_to_obj = autolab_core.RigidTransform(rot_obj, pose[0], 'world', 'obj') 
            obj_to_world = world_to_obj.inverse()           
            gcc.add_graspable_object(graspable, obj_to_world)

            graspables[item_id] = graspable

        return gcc, graspables


    def _convert_to_prob (self, ferrari_canny):
        """
        Converts Ferrari-Canny score to probability with the top being
        MAX_PROB at FC_90_THRESHOLD. 

        Prob scales from 0 -> FC_90_THRESHOLD linearly.
        """
        fc_adjusted = min (ferrari_canny, FC_90_THRESHOLD)
        return (fc_adjusted / FC_90_THRESHOLD) * MAX_PROB

    def _check_total_valid (self, state):
        """
        For debug only
        """
        print "num items" + str(len(state['poses']))
        gcc, graspables = self._create_grasp_collision_checker (state)
        item_poses = state['poses'].copy () #shallow copy
        valid_grasps = []
        valid_metrics = []

        for item_id, pose in item_poses.iteritems ():
            name = state['item_names'][item_id]
            grasps, metrics = self.dn.get_grasps (name, GRIPPER_NAME, GRASP_METRIC)
            for idx in reversed (range(len(grasps))):
                if (not self._is_collision_free (gcc, name,
                                            graspables[item_id], 
                                            grasps[idx], 
                                            item_poses[item_id])):
                    del grasps[idx]
                    del metrics[idx]
            print str(len(grasps)) + " valid grasps for item " + state['item_names'][item_id] + " z pos: " + str(item_poses[item_id][0][2])

            valid_grasps += grasps
            valid_metrics += metrics

        assert (len(valid_grasps) == len(valid_metrics))
        print "total valid grasps: " + str (len (valid_grasps))


    def find (self, state):
        """
        Finds actions for the state
        """
        #self._check_total_valid (state) #Debug only

        min_return_actions = min (MIN_RETURN_ACTIONS, len(state['poses']))

        gcc, graspables = self._create_grasp_collision_checker (state)

        item_poses = state['poses'].copy () #shallow copy
        item_actions = {}

        return_actions = []

        for item_id, pose in item_poses.iteritems ():
            name = state['item_names'][item_id]
            grasps, metrics = self.dn.get_grasps (name, GRIPPER_NAME, GRASP_METRIC)

            #TODO: slow, should cache these grasps instead of pruning all the time
            for idx in reversed (range (len(grasps))):
                if metrics[idx] < FC_PRUNE_THRESHOLD:
                    del metrics[idx]
                    del grasps[idx]

            added = False
            item_actions[item_id] = {"grasps": grasps,
                                     "metrics": metrics}

            checked_idxes = []

            for i in range (ACTION_COLLISION_CHECK_MAX_SOFT):
                idx = (i * ACTION_SKIP_RATE) % len (grasps)
                checked_idxes.append (idx)
                if self._is_collision_free (gcc, state['item_names'][item_id],
                                            graspables[item_id], 
                                            grasps[idx], 
                                            item_poses[item_id]):
                    action = Action (item_id, name, grasps[idx], 
                                     self._convert_to_prob (metrics[idx]))
                    return_actions.append (action)
                    break

            #Remove checked idxes
            for idx in reversed (checked_idxes):
                del grasps[idx]
                del metrics[idx]


        if len (return_actions) >= MIN_RETURN_ACTIONS:
            return return_actions #early return

        for action in return_actions:
            del item_poses[action.item_id]


        unlikely_item_poses = item_poses

        assert len (unlikely_item_poses) + len (return_actions) == len (state['poses'])

        give_up_iter = 0
        while (len (return_actions) < MIN_RETURN_ACTIONS 
               and len (unlikely_item_poses) > 0 
               and give_up_iter < ACTION_COLLISION_CHECK_MAX_HARD):

            #print "Enter while loop to search for an action"
            for item_id, pose in unlikely_item_poses.iteritems():
                grasps = item_actions[item_id]["grasps"]
                metrics = item_actions[item_id]["metrics"]
                assert len (grasps) == len (metrics)

                item_idx = random.randint(0, len (grasps) - 1)
                if self._is_collision_free (gcc, state['item_names'][item_id],
                                            graspables[item_id], 
                                            grasps[item_idx], 
                                            item_poses[item_id]):
                    action = Action (item_id, name, grasps[item_idx], 
                                     self._convert_to_prob (metrics[item_idx]))
                    return_actions.append (action)
                else:
                    del grasps[item_idx]
                    del metrics[item_idx]

            for action in return_actions:
                try:
                    del unlikely_item_poses[action.item_id]
                except KeyError:
                    pass

            give_up_iter += 1

        #print "Exit while loop to search for an action"
        #print "Actions Length: " + str(len(return_actions))
        

        return return_actions

   
if __name__ == "__main__":
    state = {'item_names': {5L: '437678d4bc6be981c8724d5673a063a6', 
                            6L: '5c5a9db9d4ff156a1de495b75c95e5ad', 
                            7L: 'e6f95cfb9825c0f65070edcf21eb751c'}, 
            'poses': {5L: ([-0.03550167, -0.02851296,  0.08055547], 
                           [ 0.30126049,  0.05754684, -0.73074917,  0.60986567]), 
                      6L: ([-0.01447178, -0.04350919,  0.01318321], 
                           [-3.72947186e-01,  5.19353120e-01,  7.52543106e-04,  7.68883715e-01]), 
                      7L: ([-0.00907455, -0.00746043,  0.16875964],
                           [-0.67121325, -0.15184073, -0.01838945,  0.72531303])},
            'item_ids': {'e6f95cfb9825c0f65070edcf21eb751c': 7L, 
                         '5c5a9db9d4ff156a1de495b75c95e5ad': 6L, 
                         '437678d4bc6be981c8724d5673a063a6': 5L}}
    af = ActionFinder()
    actions = af.find (state)
    assert actions is not None
    assert len(actions) > 0
    print "Basic test pass!"


'''

 def _get_actions(self, state, use_all_actions = False):
        """Get all actions given the current state.

        This can only be used in mdp mode.
        """
        if self.pomdp:
            return []
        item_names = state['item_names']

        poses = state['poses']
        actions = []

        for item_id, pose in poses.items():
            name = item_names[item_id]
            grasps, metrics = self.dn.get_grasps(name, GRIPPER_NAME, GRASP_METRIC)
            for idx, g in enumerate (grasps):
                if not use_all_actions and idx >= DEFAULT_NUM_GRASPS_PER_ITEM:
                    break
                actions.append(Action(item_id, name, grasps[idx], metrics[idx]))

        actions = self.check_collisions(state, actions)

        return actions

        def _check_collisions(self, state, actions):
        """
        Filter the provided actions for actions
        which don't cause collisions.
        """
        if self.pomdp:
            return []

        for idx in reversed(range(len(actions))):
            action = actions[idx]
            pose = poses[action.item_id]
            rot_obj = autolab_core.RigidTransform.rotation_from_quaternion(pose[1])
            rot_grip = autolab_core.RigidTransform.rotation_from_quaternion(self.gripper_pose[1])
            world_to_obj = autolab_core.RigidTransform(rot_obj, pose[0], 'world', 'obj')
            world_to_grip = autolab_core.RigidTransform(rot_grip, self.gripper_pose[0],
                                                        'world', 'gripper')
            obj_to_world = world_to_obj.inverse()
            obj_to_grip = world_to_grip.dot(obj_to_world)
            # now have RigidTransform of gripper wrt object, so can pass to collision check

            gcc.set_graspable_object(graspables[action.item_id])
            grasp_collide = gcc.grasp_in_collision(obj_to_grip.inverse(), action.item_name)
            approach_collide = gcc.collides_along_approach(
                action.grasp, self.cc_approach_dist, self.cc_delta_approach, action.item_name
            )
            if grasp_collide or approach_collide:
                del actions[idx]

        return actions

'''
