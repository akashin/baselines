import argparse
import gym
import numpy as np
import os

import ppaquette_gym_doom
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
from ppaquette_gym_doom.wrappers.observation_space import SetResolution

from baselines.qdqn.models import linear_model, conv_model
from baselines.qdqn.workers import Config
from baselines.qdqn.wrappers import ExternalProcess
from baselines.qdqn.train import train_qdqn


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
        img = img.astype('int8')
        # img = img.astype('float32') / 255.
        # img = np.squeeze(img)
        return img


class ScaleRewardEnv(gym.RewardWrapper):
    def __init__(self, env, scale):
        super(ScaleRewardEnv, self).__init__(env)
        self.scale = scale

    def _reward(self, reward):
        return reward / self.scale


def make_env(env_name, seed):
    def _make_env():
        env_spec = gym.spec(env_name)
        env_spec.id = env_name.split('/')[1]
        env = env_spec.make()
        env = SetResolution('160x120')(env)
        env = PreprocessImage((SkipWrapper(4)(ToDiscrete("minimal")(env))),
                width=80, height=80)

        scale = 1.0
        if 'DoomBasic' in env_name:
            scale = 400.0

        return ScaleRewardEnv(env, scale)

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
    parser.add_argument('--target_update_frequency', help='Target update frequency', type=int, default=500)
    parser.add_argument('--num_iterations', help='Number of iterations', type=int, default=1e5)
    parser.add_argument('--env_name', help='Env name', type=str, default='CartPole-v0')
    parser.add_argument('--cleanup', help='Should cleanup before start?', type=bool, default=False)
    args = parser.parse_args()

    config = Config()
    config.batch_size = args.batch_size
    config.actor_count = args.actor_count
    config.tf_thread_count = args.tf_thread_count
    config.learning_rate = args.learning_rate
    config.num_iterations = args.num_iterations
    config.queue_capacity = 2 ** 17
    config.exploration_schedule = "linear"
    config.target_update_frequency = args.target_update_frequency

    ALGO = "QDQN"
    env_dir = "./results/{}".format(escaped(args.env_name))
    log_dir = os.path.join(env_dir, "{}{}".format(ALGO, config))

    print("Running training with arguments: {} and log_dir: {}".format(args, log_dir))

    make_env_fn = lambda seed: make_env(args.env_name, seed)

    train_qdqn(config=config, log_dir=log_dir, make_env=make_env_fn, model=conv_model,
            cleanup=args.cleanup)


if __name__ == '__main__':
    main()
