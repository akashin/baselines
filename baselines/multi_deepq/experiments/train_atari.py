import argparse
import gym
import numpy as np
import tensorflow as tf

import os

import threading

from baselines import logger

import baselines.common.tf_util as U
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame

from baselines.multi_deepq.models import linear_model, conv_model
from baselines.multi_deepq.workers import Worker, StupidWorker, Config
from baselines.multi_deepq.wrappers import ExternalProcess


from gym.wrappers import SkipWrapper
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box


def make_env(env_name, seed):
    env = gym.make(env_name)
    env.seed(seed)
    return wrap_dqn(env)

def escaped(env_name):
    return env_name.replace('/', '.')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--worker_count', help='Worker count', type=int, default=1)
    parser.add_argument('--tf_thread_count', help='TensorFlow threads count', type=int, default=8)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=5e-4)
    parser.add_argument('--train_frequency', help='Train frequency', type=int, default=2)
    parser.add_argument('--target_update_frequency', help='Target update frequency', type=int, default=500)
    parser.add_argument('--num_iterations', help='Number of iterations', type=int, default=1e3)
    parser.add_argument('--env_name', help='Env name', type=str, default='CartPole-v0')
    args = parser.parse_args()

    config = Config()
    config.batch_size = args.batch_size
    config.worker_count = args.worker_count
    config.tf_thread_count = args.tf_thread_count
    config.learning_rate = args.learning_rate
    config.train_frequency = args.train_frequency
    config.num_iterations = args.num_iterations
    config.target_update_frequency = args.target_update_frequency

    np.random.seed(42 + config.seed)
    tf.set_random_seed(7 + config.seed)

    ALGO = "DQN"
    env_dir = "./results/{}".format(escaped(args.env_name))
    log_dir = os.path.join(env_dir, "{}{}".format(ALGO, config))
    logger.configure(dir=log_dir)
    print("Running training with arguments: {} and log_dir: {}".format(args, log_dir))

    coord = tf.train.Coordinator()

    model = conv_model

    workers = []
    for i in range(args.worker_count):
        workers.append(Worker(i == 0, make_env(args.env_name, i + config.seed), model, config, should_render=False))
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
