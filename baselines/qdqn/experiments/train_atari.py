import argparse
import gym
import numpy as np
import tensorflow as tf
import os

import threading

from baselines import logger

import baselines.common.tf_util as U
from baselines.common.atari_wrappers import WarpFrame
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame


import ppaquette_gym_doom
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
from ppaquette_gym_doom.wrappers.observation_space import SetResolution

from baselines.qdqn.models import linear_model, conv_model
from baselines.qdqn.workers import Actor, Learner, StupidWorker, Config
from baselines.qdqn.wrappers import ExternalProcess

from baselines.common.misc_util import SimpleMonitor


from gym.wrappers import SkipWrapper
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box


def make_env(env_name, seed):
    env = gym.make(env_name)
    env.seed(seed)
    return ScaledFloatFrame(wrap_dqn(env))


def escaped(env_name):
    return env_name.replace('/', '.')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--actor_count', help='Worker count', type=int, default=1)
    parser.add_argument('--tf_thread_count', help='TensorFlow threads count', type=int, default=8)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=5e-4)
    parser.add_argument('--env_name', help='Env name', type=str, default='CartPole-v0')
    args = parser.parse_args()

    config = Config()
    config.batch_size = args.batch_size
    config.actor_count = args.actor_count
    config.tf_thread_count = args.tf_thread_count
    config.learning_rate = args.learning_rate

    ALGO = "QDQN"

    np.random.seed(42)
    tf.set_random_seed(7)

    log_dir = "./results/{}_{}{}".format(ALGO, escaped(args.env_name), config)

    def create_learner_logger(base_dir):
        log_dir = os.path.join(base_dir, "learner")
        return logger.Logger(log_dir,
                [logger.make_output_format(f, log_dir) for f in logger.LOG_OUTPUT_FORMATS]
                )

    def create_actor_logger(base_dir, index):
        log_dir = os.path.join(base_dir, "actor_{}".format(index))
        return logger.Logger(log_dir,
                [logger.make_output_format(f, log_dir) for f in logger.LOG_OUTPUT_FORMATS]
                )

    print("Running training with arguments: {} and log_dir: {}".format(args, log_dir))

    coord = tf.train.Coordinator()

    model = conv_model

    env = make_env(args.env_name, 666)
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()

    capacity = 2 ** 20 / 4
    # min_after_dequeue = 2 ** 10

    queue = tf.PriorityQueue(capacity=capacity,
            types=[tf.float32, tf.int32, tf.float32, tf.float32, tf.float32],
            shapes=[observation_space.shape, action_space.shape, [], observation_space.shape, []])

    workers = []
    workers.append(Learner(observation_space, action_space, model, queue, config, create_learner_logger(log_dir)))
    for i in range(args.actor_count):
        workers.append(Actor(i, i == 0, make_env(args.env_name, i), model, queue, config, create_actor_logger(log_dir, i),
            should_render=False))
        # workers.append(StupidWorker(i == 0, make_env(args.env_name, i), model))

    with U.make_session(args.tf_thread_count) as session:
        U.initialize(session=session)

        threads = []
        for worker in workers:
            worker_fn = lambda: worker.run(session, coord)
            thread = threading.Thread(target=worker_fn)
            thread.start()
            threads.append(thread)

        coord.join(threads)


if __name__ == '__main__':
    main()
