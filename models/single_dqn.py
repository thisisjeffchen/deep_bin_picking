import tensorflow as tf

class SingleDQN():
    def __init__(self, env, flags, logger=None):
        self.flags = flags
        #TODO: fix rest of this
        pass
    
    def features_len ():
        return 1000 #TODO: make this work, there's some of this in ReplayBuffer
                            
    def build(self):
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        s = self.process_state(self.s)
        self.q = self.get_q_values_op(s, scope="q", reuse=False)

        # compute Q values of next state
        sp = self.process_state(self.sp)
        self.target_q = self.get_q_values_op(sp, scope="target_q", reuse=False)

        # add update operator for target network
        self.add_update_target_op("q", "target_q")

        # add square loss
        self.add_loss_op(self.q, self.target_q)

        # add optmizer for the main networks
        self.add_optimizer_op("q")
        
    def process_state(state):
        """
        We're already storing state values as floats, so no need to do anything to state 
        """
        return state
        
    def add_placeholders_op(self):
        fl = self.features_len ()
        
        self.states = tf.placeholder (tf.float32, shape = (None, fl), name = "states")
        raise NotImplementedError
                                  
        
      
    def add_update_target_op(self, q_scope, target_q_scope):
        raise NotImplementedError

    def add_loss_op(self, q, target_q):
        raise NotImplementedError


      
    def add_optimizer_op(self, scope):
        raise NotImplementedError

      
  
    
    #TODO: this really needs some work, thinking required
    def get_q_values_op(self, state, scope, reuse=False):
        raise NotImplementedError


    def save(self):
        raise NotImplementedError

      
    def get_action(self, state):
        raise NotImplementedError


    def init_averages(self):
        raise NotImplementedError

      
    def update_averages(self, rewards, max_q_values, q_values, scores_eval):
        raise NotImplementedError

      
    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # initialize replay buffer and variables
        replay_buffer = RobotReplayBuffer(self.flags.buffer_size)
        rewards = deque(maxlen=self.flags.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = last_record = 0 # time control of nb of steps
        scores_eval = [] # list of scores computed at iteration time
        scores_eval += [self.evaluate()]
        
        prog = Progbar(target=self.flags.nsteps_train)

        # interact with environment
        while t < self.flags.nsteps_train:
            total_reward = 0
            state = self.env.reset()
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                if self.flags.render_train: self.env.render()
                # replay memory stuff
                idx      = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                # chose action according to current Q and exploration
                best_action, q_values = self.get_best_action(q_input)
                action                = exp_schedule.get_action(best_action)

                # store q values
                max_q_values.append(max(q_values))
                q_values += list(q_values)

                # perform action in env
                new_state, reward, done, info = self.env.step(action)

                # store the transition
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # perform a training step
                loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)

                # logging stuff
                if ((t > self.flags.learning_start) and (t % self.flags.log_freq == 0) and
                   (t % self.flags.learning_freq == 0)):
                    self.update_averages(rewards, max_q_values, q_values, scores_eval)
                    exp_schedule.update(t)
                    lr_schedule.update(t)
                    if len(rewards) > 0:
                        prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg R", self.avg_reward), 
                                        ("Max R", np.max(rewards)), ("eps", exp_schedule.epsilon), 
                                        ("Grads", grad_eval), ("Max Q", self.max_q), 
                                        ("lr", lr_schedule.epsilon)])

                elif (t < self.flags.learning_start) and (t % self.flags.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(t, 
                                                        self.flags.learning_start))
                    sys.stdout.flush()

                # count reward
                total_reward += reward
                if done or t >= self.flags.nsteps_train:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)          

            if (t > self.flags.learning_start) and (last_eval > self.flags.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                scores_eval += [self.evaluate()]

            if (t > self.flags.learning_start) and self.flags.record and (last_record > self.flags.record_freq):
                self.logger.info("Recording...")
                last_record =0
                self.record()

        # last words
        self.logger.info("- Training done.")
        self.save()
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.flags.plot_output)


    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if (t > self.flags.learning_start and t % self.flags.learning_freq == 0):
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % self.flags.target_update_freq == 0:
            self.update_target_params()
            
        # occasionaly save the weights
        if (t % self.flags.saving_freq == 0):
            self.save()

        return loss_eval, grad_eval


    def evaluate(self, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.flags.num_episodes_test

        if env is None:
            env = self.env

        # replay memory to play
        replay_buffer = RobotReplayBuffer(self.flags.buffer_size, self.flags.state_history)
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            state = env.encode_state(state)
            while True:
                if self.flags.render_test: env.render()

                # store last state in buffer
                idx     = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                action = self.get_action(q_input)

                # perform action in env
                new_state, reward, done, info = env.step(action)
                new_state = env.encode_state(new_state)

                # store in replay memory
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # count reward
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)     

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            self.logger.info(msg)

        return avg_reward


    def run(self, exp_schedule, lr_schedule):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()

        # record one game at the beginning
        if self.flags.record:
            self.record()

        # model
        self.train(exp_schedule, lr_schedule)

        # record one game at the end
        # if self.config.record:
        #     self.record()

  
    def record(self):
        """
        Re create an env and record a video for one episode
        """
        pass
 
if __name__ == "__main__":
    sdqn = SingleDQN (None, None)
    print "Init works!"