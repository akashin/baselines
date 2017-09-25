#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_disc import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.acktr.policies import CnnPolicy


import ppaquette_gym_doom
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
from gym.spaces.box import Box
from gym.wrappers import SkipWrapper
from scipy.misc import imresize

class PreprocessImage(gym.ObservationWrapper):
    def __init__(self, env, height=64, width=64, grayscale=True,
                 crop=lambda img: img):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop

        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0, 255, [height, width, n_colors])

    def _observation(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        # img = np.transpose(img, (2, 0, 1))  # reshape from (h,w,colors) to (colors,h,w)
        # img = img.astype('float32') / 255.
        # img = np.squeeze(img)
        return img


class ScaleRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        return reward / 400.0



def train(env_id, num_frames, seed, num_cpu):
    num_timesteps = int(num_frames / 4 * 1.1) 
    def make_env(rank):
        def _thunk():
            env_spec = gym.spec('ppaquette/DoomBasic-v0')
            env_spec.id = 'DoomBasic-v0'
            env = env_spec.make()
            env.seed(seed + rank)
            env = PreprocessImage((SkipWrapper(4)(ToDiscrete("minimal")(env))))
            if logger.get_dir():
                env = bench.Monitor(env, os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return ScaleRewardEnv(env)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    policy_fn = CnnPolicy
    learn(policy_fn, env, seed, total_timesteps=num_timesteps, nprocs=num_cpu, nstack=1)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_cpu', help='Number of workers', type=int, default=8)
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip', type=int, default=40)
    args = parser.parse_args()
    train(args.env, num_frames=1e6 * args.million_frames, seed=args.seed, num_cpu=args.num_cpu)


if __name__ == '__main__':
    main()
