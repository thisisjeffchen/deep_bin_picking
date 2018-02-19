class crate ()
  reset ()
       #reset simulation, sample possible initial states and set the simulator to one such initial state
       #set random seed to get same results
      return obs (MDP case, obs = state, but POMDP case, must still calculate belief state from obs)
	
  step (action)
      #action has to be gripper pose (gripper center position as x,y,z and gripper orientation as quaternion) & object id. in global frame
     # determine whether move succeeded: call dex-net and get probability of grasp success
     #if move succeeded, reward = 1 and delete object, else return 0
     #compute terminal: are there still objects left on the table?
      return obs, reward, terminal

  check_collisions (actions)
        returns pruned_actions #collisions removed
