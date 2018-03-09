import tensorflow as tf

#TODO: make all flags here, it will get passed into our core SingleDQN
tf.app.flags.DEFINE_integer ("nsteps_train", 5000000, "")
tf.app.flags.DEFINE_integer ("batch_size", 32, "")
tf.app.flags.DEFINE_integer ("buffer_size", 1000000, "")
tf.app.flags.DEFINE_integer ("target_update_freq", 10000, "")
tf.app.flags.DEFINE_float   ("gamma", 0.99, "")
tf.app.flags.DEFINE_integer ("learning_freq", 4, "")
tf.app.flags.DEFINE_integer ("skip_frame", 4, "")
tf.app.flags.DEFINE_float   ("lr_begin", 0.00025, "")
tf.app.flags.DEFINE_float   ("lr_end", 0.00005, "")
tf.app.flags.DEFINE_integer ("lr_nsteps", nsteps_train/2, "") #TODO: make this work
tf.app.flags.DEFINE_integer ("eps_begin", 1, "")
tf.app.flags.DEFINE_float   ("eps_end", 0.1, "")
tf.app.flags.DEFINE_integer ("eps_nsteps", 1000000, "")
tf.app.flags.DEFINE_integer ("learning_start", 50000, "")

#TODO: this is not hooked up to the save dir
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")

#TODO: grab other needed flags from HW2 base code

FLAGS = tf.app.flags.FLAGS


def main(unused_argv):
 	env = gym.make(FLAGS.env_name) #TODO CHANGE, and make work w/ our env
    
    # exploration strategy
    exp_schedule = LinearExploration(env, FLAGS.eps_begin, 
            FLAGS.eps_end, FLAGS.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(FLAGS.lr_begin, FLAGS.lr_end,
            FLAGS.lr_nsteps)

    # train model
    model = SingleDQN(env, FLAGS)
    model.run(exp_schedule, lr_schedule)


if __name__ == "__main__":
    tf.app.run()