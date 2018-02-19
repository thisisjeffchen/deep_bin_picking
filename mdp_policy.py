class mdp_policy ()
get_action(state)
    #first call dex-net to get one best grasp per object
    # call pruned_actions = mdp.check_collisions
    Baseline algorithm: Greedy policy for now (this would ignore our rewards or values)
    Learned algorithm: some sort of RL algorithm
    
    #to decide: which is the best actual algo to use

run ()
  obs = env.reset ()
  rewards = 0
  steps = 0
  while (not terminal)
      action = gripper_policy.get_action (obs)
      obs, r, terminal = env.step (action)
      rewards += gamma^steps * r 		# gamma = 0.9
      steps += 1
      #update gripper_policy if needed
