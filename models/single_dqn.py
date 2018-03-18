import tensorflow as tf
import collections
import numpy as np
import os
import pdb

from utils.general import get_logger, Progbar, export_plot
from robot_replay_buffer import RobotReplayBuffer
from collections import deque

NN_HIDDEN_1 = 50
NN_HIDDEN_2 = 10

NN_DEEP_HIDDEN = 50

class SingleDQN():
    def __init__(self, env, flags, logger=None):
        self.flags = flags
        self.env = env

        self.build ()

        # create tf session
        self.sess = tf.Session()

        # tensorboard stuff
        self.add_summary()
        # logger depends on the dir being there
        if logger is None:
            log_path = self.flags.train_dir + "/log.txt"
            self.logger = get_logger(log_path)

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # synchronise q and target_q networks
        self.sess.run(self.update_target_op)

        # for saving networks weights
        self.saver = tf.train.Saver()


    def add_summary(self):
        """
        Tensorboard stuff
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.avg_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="avg_q")
        self.max_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="max_q")
        self.std_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="std_q")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads norm", self.grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std Reward", self.std_reward_placeholder)

        tf.summary.scalar("Avg Q", self.avg_q_placeholder)
        tf.summary.scalar("Max Q", self.max_q_placeholder)
        tf.summary.scalar("Std Q", self.std_q_placeholder)

        tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)

        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.flags.train_dir,
                                                self.sess.graph)
    
    def features_len (self):
        """
        Flattened observation len
        """
        s = self.env.get_encoded_observation_shape ()
        return s[0] * s[1]

    def actions_len (self):
        """
        How long each action is
        """
        return self.env.get_action_dims ()

    def actions_choices_len (self):
        """
        How many choices are fed into the graph. We hard code this else it 
        makes running through the graph difficult.
        """
        return self.env.get_action_choices_max ()
                            
    def build(self):
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        self.q_prior= self.get_q_values_op(self.states, self.actions_taken,
                                      scope="q", reuse=False)

        # compute Q values of next state
        states_prime_tiled = tf.tile (self.states_prime, [self.actions_choices_len (), 1])

        s = tf.shape (self.action_choices)
        action_choices_tiled =  tf.reshape (self.action_choices, [-1, self.actions_len ()])

        target_q_tiled = self.get_q_values_op (states_prime_tiled,
                                               action_choices_tiled,
                                               scope = "target_q",
                                               reuse = False)

        self.target_q = tf.reshape (target_q_tiled, shape = (s[0], s[1])) #(batch, actions_choices_len)

        # add update operator for target network
        self.add_update_target_op("q", "target_q")

        # add square loss
        self.add_loss_op(self.q_prior, self.target_q)

        # add optmizer for the main networks
        self.add_optimizer_op("q")
        
        
    def add_placeholders_op(self):
        """
        states: batch x states_len
        actions_taken: batch x action_len
        reward: batch x 1
        states_prime: batchx states_len
        action_choices: batch x action_choices_len x action_len
        """
        fl = self.features_len ()
        al = self.actions_len ()
        acl = self.actions_choices_len ()
    
        self.states = tf.placeholder (tf.float32, shape = (None, fl), 
                                      name = "states")
        self.states_prime = tf.placeholder (tf.float32, shape = (None, fl),
                                            name = "states_prime")

        self.actions_taken = tf.placeholder (tf.float32, shape = (None, al), 
                                       name = "actions_taken")

        self.action_choices = tf.placeholder (tf.float32, 
                                              shape = (None, acl, al),
                                              name = "action_choices")
        self.action_choices_mask = tf.placeholder (tf.bool,
                                                    shape = (None, acl),
                                                    name = "action_choices_mask")

        self.rewards = tf.placeholder (tf.float32, shape = (None),
                                       name = "rewards")
        self.done_mask = tf.placeholder (tf.float32, shape = (None), 
                                         name = "done_mask")
        self.lr = tf.placeholder (tf.float32, shape = (), name = "lr")
                                  
      
    def add_update_target_op(self, q_scope, target_q_scope):
        Q_vars = tf.get_collection (tf.GraphKeys.TRAINABLE_VARIABLES, 
                                   scope = q_scope)
        target_Q_vars = tf.get_collection (tf.GraphKeys.TRAINABLE_VARIABLES, 
                                           scope = target_q_scope)

        ops = []
        for i in range (len (target_Q_vars)):
            ops.append (tf.assign (target_Q_vars[i], Q_vars[i]))

        self.update_target_op = tf.group (*ops, name = "update_target_op")

    def add_loss_op(self, q_prior, target_q):
        target_q_masked = tf.boolean_mask (target_q, self.action_choices_mask)
        target_q_max = tf.reduce_max (target_q_masked, axis = 0) #(batch)


        q_samp = self.rewards + (1 - tf.cast (self.done_mask, tf.float32)) * (
                         tf.cast(self.flags.gamma, tf.float32)
                         * target_q_max)

        self.loss = tf.reduce_mean (tf.square (q_samp - q_prior), name = "loss")
      
    def add_optimizer_op(self, scope):
        opt = tf.train.AdamOptimizer (learning_rate = self.lr)
        variables = tf.get_collection (tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope)
        grads_and_vars = opt.compute_gradients (self.loss, variables)

        if self.flags.grad_clip:
            grads_and_vars = [(tf.clip_by_norm (gv[0], self.flags.clip_val), gv[1]) for gv in grads_and_vars]

        self.train_op = opt.apply_gradients (grads_and_vars, name = "train_op")

        self.grad_norm = tf.global_norm ([gv[0] for gv in grads_and_vars], name = "grad_norm")
        writer = tf.summary.FileWriter(self.flags.train_dir, graph=tf.get_default_graph())  
    
    def get_q_values_op(self, states, actions, scope, reuse=False):
        """
        States batch x features_len
        Actions batch x actions_len
        Returns output depending on model selection
        """

        inputs = tf.concat ([states, actions], 1)
        out = self.rewards #placeholder so init doesn't complain. 


        with tf.variable_scope (scope, reuse = reuse):
            if self.flags.model == "linear":
                out = tf.contrib.layers.fully_connected (inputs, 1,
                                                         activation_fn = None)
            elif self.flags.model == "nn":
                layer_1 = tf.contrib.layers.fully_connected (inputs, NN_HIDDEN_1) #ReLU by default
                layer_2 = tf.contrib.layers.fully_connected (layer_1, NN_HIDDEN_2) #ReLU by default
                out = tf.contrib.layers.fully_connected (layer_2, 1, activation_fn = None)
            elif self.flags.model == "nn_deep":
                layer_1 = tf.contrib.layers.fully_connected (inputs, NN_DEEP_HIDDEN)
                layer_2 = tf.contrib.layers.fully_connected (layer_1, NN_DEEP_HIDDEN)
                layer_3 = tf.contrib.layers.fully_connected (layer_2, NN_DEEP_HIDDEN)
                layer_4 = tf.contrib.layers.fully_connected (layer_3, NN_DEEP_HIDDEN)

                out = tf.contrib.layers.fully_connected (layer_4, 1, activation_fn = None)


            else:
                raise NameError ("Model flag not recognized") 
        return out

    def save(self):
        """
        Saves session
        """
        self.saver.save(self.sess, self.flags.train_dir)

      
    def get_best_action(self, state, action_choices):


        fd = {
            self.states: np.tile (state, [len(action_choices), 1]),
            self.actions_taken: action_choices
        }

        q_values = self.sess.run (self.q_prior, feed_dict = fd)

        return np.argmax (q_values), q_values


    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """

        obs_batch, act_choices_batch, act_choices_mask, act_batch, rew_batch, done_mask_batch, next_obs_batch = replay_buffer.sample(self.flags.batch_size)


        fd = {
            # inputs
            self.states: obs_batch,
            self.states_prime: next_obs_batch,
            self.actions_taken: act_batch,
            self.action_choices: act_choices_batch,
            self.action_choices_mask: act_choices_mask,
            self.rewards: rew_batch,
            self.done_mask: done_mask_batch,
            self.lr: lr,
            # extra info
            self.avg_reward_placeholder: self.avg_reward,
            self.max_reward_placeholder: self.max_reward,
            self.std_reward_placeholder: self.std_reward,
            self.avg_q_placeholder: self.avg_q,
            self.max_q_placeholder: self.max_q,
            self.std_q_placeholder: self.std_q,
            self.eval_reward_placeholder: self.eval_reward,
        }


        loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss, self.grad_norm,
                                                 self.merged, self.train_op], feed_dict=fd)


        # tensorboard stuff
        self.file_writer.add_summary(summary, t)

        return loss_eval, grad_norm_eval
    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.sess.run(self.update_target_op)


    def init_averages(self):
        """
        Defines extra attributes for tensorboard
        """
        self.avg_reward = -100.
        self.max_reward = -100.
        self.std_reward = 0

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0
        
        self.eval_reward = -100.

    def update_averages(self, rewards, max_q_values, q_values, scores_eval):
        """
        Update the averages

        Args:
            rewards: deque
            max_q_values: deque
            q_values: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q      = np.mean(max_q_values)
        self.avg_q      = np.mean(q_values)
        self.std_q      = np.sqrt(np.var(q_values) / len(q_values))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]
      
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
        rewards = deque(maxlen=self.flags.running_avg_size)
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
            s = 0

            while True:
                t += 1
                last_eval += 1
                last_record += 1
                
                action_choices = self.env.get_current_candidate_actions ()
                if len(action_choices) == 0:
                    done = True
                    break

                # replay memory stuff
                encoded_actions, encoded_actions_mask = self.env.encode_action_choices (action_choices, state)
                encoded_state = self.env.encode_state (state)

                idx = replay_buffer.store_frame(encoded_state,
                                                encoded_actions, 
                                                encoded_actions_mask)

                best_action_idx, q_values = self.get_best_action (encoded_state, encoded_actions[0:len(action_choices)])
                action_idx = exp_schedule.get_action_idx(best_action_idx, action_choices)
                action = action_choices[action_idx]


                max_q_values.append(max(q_values))
                q_values += list(q_values)

                state, reward, done = self.env.step(action)

                replay_buffer.store_effect(idx, encoded_actions[action_idx], reward, done)

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
                    print ("\rPopulating the memory {}/{}...".format(t, self.flags.learning_start))

                # count reward
                total_reward += reward * (self.flags.gamma ** s)
                if done or t >= self.flags.nsteps_train:
                    break
                s+=1

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
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            s = 0
            while True:
                action_choices = self.env.get_current_candidate_actions ()
                if len(action_choices) == 0:
                    done = True
                    break

                encoded_actions, encoded_actions_mask = self.env.encode_action_choices (action_choices, state)
                encoded_state = self.env.encode_state (state)

                best_action_idx, q_values = self.get_best_action (encoded_state, encoded_actions[0:len(action_choices)])

                action = action_choices[best_action_idx]

                state, reward, done = self.env.step(action)

                # count reward
                total_reward += reward * (self.flags.gamma ** s)
                if done:
                    break
                s += 1

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

        # model
        if self.flags.record:
             self.record()
        
        self.train(exp_schedule, lr_schedule)

        if self.flags.record:
             self.record()

  
    def record(self):
        """
        Re create an env and record a video for one episode
        """
        #TODO: figure out howto record
        pass

 
if __name__ == "__main__":
    Flags = collections.namedtuple('Flags', ['model', 'gamma', 'grad_clip', 'train_dir'])
    flags = Flags (model = "linear", gamma = 0.9, grad_clip = False, train_dir = "test")


    sdqn = SingleDQN (None, flags)
    print "Init works!"
