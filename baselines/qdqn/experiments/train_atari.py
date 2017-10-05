import argparse
import gym
import os

import baselines.common.tf_util as U
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame

from baselines.qdqn.models import linear_model, conv_model
from baselines.qdqn.workers import Config

from baselines.qdqn.train import train_qdqn

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
    parser.add_argument('--actor_count', help='Worker count', type=int, default=1)
    parser.add_argument('--tf_thread_count', help='TensorFlow threads count', type=int, default=8)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=5e-4)
    parser.add_argument('--num_iterations', help='Number of iterations', type=int, default=1e5)
    parser.add_argument('--env_name', help='Env name', type=str, default='CartPole-v0')
    args = parser.parse_args()

    config = Config()
    config.batch_size = args.batch_size
    config.actor_count = args.actor_count
    config.tf_thread_count = args.tf_thread_count
    config.learning_rate = args.learning_rate
    config.num_iterations = args.num_iterations
    config.queue_capacity = 2 ** 17
    config.exploration_schedule = "piecewise"

    ALGO = "QDQN"
    env_dir = "./results/{}".format(escaped(args.env_name))
    log_dir = os.path.join(env_dir, "{}{}".format(ALGO, config))

    print("Running training with arguments: {} and log_dir: {}".format(args, log_dir))

    make_env_fn = lambda seed: make_env(args.env_name, seed)

    train_qdqn(config=config, log_dir=log_dir, make_env=make_env_fn, model=conv_model)


if __name__ == '__main__':
    main()
