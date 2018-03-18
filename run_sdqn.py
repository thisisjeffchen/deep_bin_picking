import tensorflow as tf
import math
import os

from utils.general import get_logger
from env.crate import CrateMDP
from env.scene import Scene, ScenePopulator
from models.schedule import LinearExploration, LinearSchedule
from models.single_dqn import SingleDQN


EXPERIMENTS_DIR = "./experiments"


tf.app.flags.DEFINE_string  ("model", "linear", "Model used, choices are linear or nn")
tf.app.flags.DEFINE_integer ("nsteps_train", 100000, "")
tf.app.flags.DEFINE_integer ("batch_size", 50, "")
tf.app.flags.DEFINE_integer ("buffer_size", 100000, "")
tf.app.flags.DEFINE_integer ("target_update_freq", 1000, "")
tf.app.flags.DEFINE_float   ("gamma", 0.9, "")
tf.app.flags.DEFINE_integer ("learning_freq", 4, "")
tf.app.flags.DEFINE_float   ("lr_begin", 0.0025, "")
tf.app.flags.DEFINE_float   ("lr_end", 0.00005, "")
tf.app.flags.DEFINE_integer ("lr_nsteps", -1, "Initially set to -1, but later corrected below")
tf.app.flags.DEFINE_integer ("eps_begin", 1, "")
tf.app.flags.DEFINE_float   ("eps_end", 0.1, "")
tf.app.flags.DEFINE_integer ("eps_nsteps", 20000, "")
tf.app.flags.DEFINE_integer ("learning_start", 1000, "") #needs to be at least 1 episode
tf.app.flags.DEFINE_bool    ("grad_clip", False, "")
tf.app.flags.DEFINE_integer ("clip_val", 10, "")

tf.app.flags.DEFINE_integer ("num_episodes_test", 50, "")
tf.app.flags.DEFINE_integer ("running_avg_size", 250, "")
tf.app.flags.DEFINE_integer ("saving_freq", 100, "")
tf.app.flags.DEFINE_integer ("log_freq", 10, "")
tf.app.flags.DEFINE_integer ("eval_freq", 2000, "")
tf.app.flags.DEFINE_integer ("record_freq", 200, "")
tf.app.flags.DEFINE_float   ("soft_epsilon", 0.05, "")
tf.app.flags.DEFINE_string  ("train_dir", "", "defaults to experiment_name")
tf.app.flags.DEFINE_string("experiment_name", "tmp", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_bool    ("record", False, "")
tf.app.flags.DEFINE_bool    ("tiny_space", False, "Use this flag to get use a tiny action space of the z pos and the action success prob")



train_path = os.path.join(EXPERIMENTS_DIR, tf.app.flags.FLAGS.experiment_name)


def main(unused_argv):
    FLAGS = tf.app.flags.FLAGS
    FLAGS.lr_nsteps = FLAGS.nsteps_train/2
    FLAGS.train_dir = train_path

    scene = Scene (show_gui=False)
    scene_populator = ScenePopulator (scene)
    env = CrateMDP (scene, scene_populator, flags=FLAGS)

    # exploration strategy
    exp_schedule = LinearExploration (env, FLAGS.eps_begin, 
                                      FLAGS.eps_end, FLAGS.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(FLAGS.lr_begin, FLAGS.lr_end,
                                  FLAGS.lr_nsteps)

    # train model
    model = SingleDQN (env, FLAGS)
    model.run (exp_schedule, lr_schedule)


if __name__ == "__main__":
    tf.app.run()
