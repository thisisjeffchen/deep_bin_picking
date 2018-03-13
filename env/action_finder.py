import dexnet

DEX_NET_PATH = '../../dex-net/'
DB_NAME = 'dexnet_2.hdf5'
GRIPPER_NAME = 'yumi_metal_spline'
GRIPPER_REL_PATH = 'data/grippers/'
GRASP_METRIC = 'ferrari_canny'
DEFAULT_NUM_GRASPS_PER_ITEM = 3
GRIPPER_Z_POS = 1
FC_90_THRESHOLD = 0.010809481366594122
MAX_PROB = 0.9

ACTION_COLLISION_CHECK_MAX_SOFT = 3
ACTION_COLLISION_CHECK_MAX_HARD = 20
ACTION_SKIP_RATE = 11

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
        self.gripper_pose = ([0, 0, GRIPPER_Z_POS], [1, 0, 0, 0])
        self.cc_approach_dist = 1.0
        self.cc_delta_approach = 0.1    # may need tuning

 	def _check_collisions(self, state, actions):
        """Filter the provided actions for actions which don't cause collisions."""
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

    def _is_collision_free (self, graspable, grasp, pose):
    	"""
    	Makes sure a particular graspable is collision free
    	"""
    	return True



    def _create_grasp_collision_checker (self, state):
    	gcc = dexnet.grasping.GraspCollisionChecker(self.gripper)

        # Add all objects to the world frame
        item_names = state['item_names']
        poses = state['poses']
        graspables = {}
        for item_id, pose in poses.items():
            graspable = self.dn.dataset.graspable(item_names[item_id])
            gcc.add_graspable_object(graspable)
            graspables[item_id] = graspable

       	return gcc, graspables


    def _convert_to_prob (self, ferrari_canny):
    	fc_adjusted = max (ferrari_canny, FC_90_THRESHOLD)

    	return (fc_adjusted / FC_90_THRESHOLD) * MAX_PROB


	def find (self, state):
		"""
		Finds actions for the state
		"""
		gcc, graspables = _create_grasp_collision_checker (state)

		unlikely_item_poses = {}
		item_poses = {**state['poses']}
		item_actions = {}

		return_actions = []

		for item_id, pose in item_poses:
			name = state['item_names'][item_id]
			grasps, metrics = self.dn.get_grasps (name, GRIPPER_NAME, GRASP_METRIC)
			added = False
			item_actions[name] = {"grasps": grasps,
								  "metrics": metrics}

			for i in range (ACTION_COLLISION_CHECK_MAX_SOFT):
				idx = (i * ACTION_SKIP_RATE) % len (grasps)
				if self._is_collision_free (graspables[item_id], grasps[idx], item_poses[item_id]):
					action = Action (item_id, name, grasps[idx], self._convert_to_prob (metrics[idx]))
					return_actions.append (action)
					added = True



		#while #there are no actions:



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

'''

   
if __name__ == "__main__":
	#tests
