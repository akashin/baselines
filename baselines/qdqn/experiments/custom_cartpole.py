import argparse
import gym
import numpy as np
import tensorflow as tf

import threading

import baselines.common.tf_util as U

from baselines import logger

from baselines.multi_deepq.models import linear_model, conv_model
from baselines.multi_deepq.workers import Worker, StupidWorker

from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common.atari_wrappers_deprecated import wrap_dqn


def is_atari(env_name):
    return env_name.startswith('Pong')
    # return env_name in ['Pong-v0', 'PongNoFrameskip-v4', 'PongDeterministic-v4']


def make_env(env_name, seed):
    # Create the environment
    def make_gym_env():
        env = gym.make(env_name)
        env.seed(seed)
        if is_atari(env_name):
            env = wrap_deepmind(env)
        return env

    return make_gym_env()


MODE = 'training'
GAME = 'cartpole'


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--worker_count', help='Batch size', type=int, default=1)
    parser.add_argument('--tf_thread_count', help='TensorFlow threads count', type=int, default=1)
    parser.add_argument('--env_name', help='Batch size', type=str, default='CartPole-v0')
    args = parser.parse_args()

    np.random.seed(42)
    tf.set_random_seed(7)

    log_dir = "./results/multi_deepq_{}_{}_{}_batch_size_{}_worker_count_{}_tf_thread_count_{}".format(
            GAME, MODE, args.env_name, args.batch_size, args.worker_count, args.tf_thread_count)
    logger.configure(dir=log_dir)
    print("Running training with arguments: {} and log_dir: {}".format(args, log_dir))

    coord = tf.train.Coordinator()

    if is_atari(args.env_name):
        model = conv_model
    else:
        model = linear_model

    workers = []
    for i in range(args.worker_count):
        workers.append(Worker(i == 0, make_env(args.env_name, i), model, args.batch_size))
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

