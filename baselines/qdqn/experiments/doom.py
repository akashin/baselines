import argparse
import gym
import numpy as np
import tensorflow as tf
import os

import threading

from baselines import logger

import baselines.common.tf_util as U
from baselines.common.atari_wrappers import WarpFrame


import ppaquette_gym_doom
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
from ppaquette_gym_doom.wrappers.observation_space import SetResolution

from baselines.qdqn.models import linear_model, conv_model
from baselines.qdqn.workers import Actor, Learner, StupidWorker, Config
from baselines.qdqn.wrappers import ExternalProcess


from gym.wrappers import SkipWrapper
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box


class PreprocessImage(ObservationWrapper):
    def __init__(self, env, height=64, width=64, grayscale=True,
            crop=lambda img: img):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop

        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [height, width, n_colors])

    def _observation(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        # img = np.transpose(img, (2, 0, 1))  # reshape from (h,w,colors) to (colors,h,w)
        img = img.astype('float32') / 255.
        # img = np.squeeze(img)
        return img



def make_env(env_name, seed):
    def _make_env():
        env_spec = gym.spec(env_name)
        env_spec.id = env_name.split('/')[1]
        env = env_spec.make()
        env = SetResolution('160x120')(env)
        env = PreprocessImage((SkipWrapper(4)(ToDiscrete("minimal")(env))),
                width=80, height=80)
        return env

    # env = ExternalProcess(_make_env)
    env = _make_env()
    return env


def escaped(env_name):
    return env_name.replace('/', '.')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--actor_count', help='Worker count', type=int, default=1)
    parser.add_argument('--tf_thread_count', help='TensorFlow threads count', type=int, default=8)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=5e-4)
    parser.add_argument('--train_frequency', help='Train frequency', type=int, default=2)
    parser.add_argument('--env_name', help='Env name', type=str, default='CartPole-v0')
    args = parser.parse_args()

    config = Config()
    config.batch_size = args.batch_size
    config.actor_count = args.actor_count
    config.tf_thread_count = args.tf_thread_count
    config.learning_rate = args.learning_rate
    config.train_frequency = args.train_frequency

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

    env = make_env(args.env_name, -1)
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()

    capacity = 2 ** 17
    min_after_dequeue = 2 ** 10

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
