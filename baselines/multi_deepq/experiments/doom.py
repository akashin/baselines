import argparse
import gym
import numpy as np
import tensorflow as tf

import threading

from baselines import logger

import baselines.common.tf_util as U
from baselines.common.atari_wrappers import WarpFrame


import ppaquette_gym_doom
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
from ppaquette_gym_doom.wrappers.observation_space import SetResolution

from baselines.multi_deepq.models import linear_model, conv_model
from baselines.multi_deepq.workers import Worker, StupidWorker, Config
from baselines.multi_deepq.wrappers import ExternalProcess


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
    parser.add_argument('--worker_count', help='Worker count', type=int, default=1)
    parser.add_argument('--tf_thread_count', help='TensorFlow threads count', type=int, default=8)
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=5e-4)
    parser.add_argument('--train_frequency', help='Train frequency', type=int, default=2)
    parser.add_argument('--env_name', help='Env name', type=str, default='CartPole-v0')
    args = parser.parse_args()

    config = Config()
    config.batch_size = args.batch_size
    config.worker_count = args.worker_count
    config.tf_thread_count = args.tf_thread_count
    config.learning_rate = args.learning_rate
    config.train_frequency = args.train_frequency

    np.random.seed(42)
    tf.set_random_seed(7)

    log_dir = "./results/DQN_{}{}".format(escaped(args.env_name), config)
    logger.configure(dir=log_dir)
    print("Running training with arguments: {} and log_dir: {}".format(args, log_dir))

    coord = tf.train.Coordinator()

    model = conv_model

    workers = []
    for i in range(args.worker_count):
        workers.append(Worker(i == 0, make_env(args.env_name, i), model, config, should_render=False))
        # workers.append(StupidWorker(i == 0, make_env(args.env_name, i), model))

    with U.make_session(args.tf_thread_count) as session:
        U.initialize(session=session)

        workers[0].run(session, coord)

        # threads = []
        # for worker in workers:
            # worker_fn = lambda: worker.run(session, coord)
            # thread = threading.Thread(target=worker_fn)
            # thread.start()
            # threads.append(thread)

        # coord.join(threads)


if __name__ == '__main__':
    main()
