import tensorflow as tf
import tensorflow.contrib.layers as layers
import itertools
import numpy as np

from tensorflow.python.client import timeline

from timeit import default_timer as timer

import baselines.common.tf_util as U

from baselines import logger

from baselines import qdqn
from baselines.qdqn.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule

from collections import namedtuple, defaultdict

import json


class Config(object):

    def __init__(self):
        self.batch_size = 64
        self.gamma = 0.99
        self.exploration_length = 50000
        self.learning_rate = 1e-4
        self.actor_count = 1
        self.tf_thread_count = 8
        self.train_frequency = 1
        self.target_update_frequency = 500 / self.batch_size
        self.params_update_frequency = 32

    def __repr__(self):
        s = ''
        for k, v in sorted(self.__dict__.items()):
            s += ',{}={}'.format(k, v)
        return s


class EventTimer(object):

    def __init__(self):
        self.times = defaultdict(float)
        self.visits = defaultdict(int)
        self.total_time = 0
        self.started = False

    def start(self):
        self.started = True
        self.start_time = timer()
        self.current_time = self.start_time

    def measure(self, label):
        if not self.started:
            return

        current_time = timer()
        self.times[label] += current_time - self.current_time
        self.visits[label] += 1
        self.current_time = current_time

    def stop(self):
        if not self.started:
            return

        self.started = False
        self.total_time += timer() - self.start_time

    def print_shares(self, logger):
        for key, value in sorted(self.times.items()):
            logger.record_tabular(key + "_share", value / self.total_time * 100.0)

    def print_averages(self, logger):
        for key, value in sorted(self.times.items()):
            logger.record_tabular(key + "_time", value * 1.0 / self.visits[key])


class Actor(object):
    def __init__(self, index, is_chief, env, model, queue, config, logger, should_render=True):
        self.config = config
        self.is_chief = is_chief
        self.env = env
        self.should_render = should_render
        self.logger = logger

        with tf.device('/cpu:0'):
            self.act, self.update_params = qdqn.build_act(
                    make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
                    q_func=model,
                    num_actions=env.action_space.n,
                    scope="actor_{}".format(index),
                    learner_scope="learner",
                    reuse=False)

            priority_ph = tf.placeholder(tf.int64, [], name="priority")
            obs_t_input = tf.placeholder(tf.float32, env.observation_space.shape, name="obs_t")
            act_t_ph = tf.placeholder(tf.int32, env.action_space.shape, name="action")
            rew_t_ph = tf.placeholder(tf.float32, [], name="reward")
            obs_tp1_input = tf.placeholder(tf.float32, env.observation_space.shape, name="obs_tp1")
            done_mask_ph = tf.placeholder(tf.float32, [], name="done")
            enqueue_op = queue.enqueue([priority_ph, obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph])
            self.enqueue = U.function([priority_ph, obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph], enqueue_op)

        self.max_iteration_count = int(self.config.exploration_length * 5.0)
        # self.max_iteration_count = 128

        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        self.exploration = LinearSchedule(schedule_timesteps=self.config.exploration_length, initial_p=1.0, final_p=0.02)

    def run(self, session, coord):
        episode_rewards = [0.0]
        obs = self.env.reset()

        start_time = timer()
        event_timer = EventTimer()
        for t in range(self.max_iteration_count):
            if t > 0 and t % 500 == 0:
                event_timer.start()
            # Take action and update exploration to the newest value
            action = self.act(obs[None], update_eps=self.exploration.value(t), session=session)[0]
            event_timer.measure('act')
            new_obs, rew, done, _ = self.env.step(action)
            event_timer.measure('step')
            # Store transition in the replay buffer.
            self.enqueue(np.random.randint(100000), obs, action, rew, new_obs, float(done), session=session)
            # self.replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = self.env.reset()
                episode_rewards.append(0)

            event_timer.measure('queue')

            # show_episode = len(episode_rewards) > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            show_episode = len(episode_rewards) % 100 == 0
            if self.should_render and self.is_chief and show_episode:
                # Show off the result
                self.env.render()

            # Update target network periodically.
            if t % self.config.params_update_frequency == 0:
                self.update_params(session=session)
                event_timer.measure('update_params')

            event_timer.stop()

            if self.is_chief and done and len(episode_rewards) % 10 == 0:
                self.logger.record_tabular("steps", t)
                self.logger.record_tabular("episodes", len(episode_rewards))
                self.logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                self.logger.record_tabular("time elapsed", timer() - start_time)
                self.logger.record_tabular("steps/s", t / (timer() - start_time))
                self.logger.record_tabular("% time spent exploring", int(100 * self.exploration.value(t)))
                event_timer.print_shares(self.logger)
                event_timer.print_averages(self.logger)
                self.logger.dump_tabular()

class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


class Learner(object):
    def __init__(self, observation_space, action_space, model, queue, config, logger):
        self.config = config
        self.queue = queue
        self.logger = logger

        queue_size_op = self.queue.size()
        self.queue_size = U.function([], queue_size_op)

        with tf.device('/cpu:0'):
            priority_ph = tf.placeholder(tf.int64, [config.batch_size], name="priority")
            obs_t_input = tf.placeholder(tf.float32, [config.batch_size] + list(observation_space.shape), name="obs_t")
            act_t_ph = tf.placeholder(tf.int32, [config.batch_size] + list(action_space.shape), name="action")
            rew_t_ph = tf.placeholder(tf.float32, [config.batch_size] + [], name="reward")
            obs_tp1_input = tf.placeholder(tf.float32, [config.batch_size] + list(observation_space.shape), name="obs_tp1")
            done_mask_ph = tf.placeholder(tf.float32, [config.batch_size] + [], name="done")
            enqueue_op = queue.enqueue_many([priority_ph, obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph])
            self.enqueue = U.function([priority_ph, obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph], enqueue_op)

            dequeue_op = self.queue.dequeue_many(config.batch_size)
            self.dequeue = U.function([], dequeue_op)

        with tf.device('/gpu:0'):
            self.act, self.train, self.update_target, self.debug = qdqn.build_train(
                    make_obs_ph=lambda name: U.BatchInput(observation_space.shape, name=name),
                    q_func=model,
                    num_actions=action_space.n,
                    gamma=config.gamma,
                    optimizer=tf.train.AdamOptimizer(learning_rate=config.learning_rate),
                    scope="learner",
                    reuse=False)

            self.max_iteration_count = int(self.config.exploration_length * 5.0)
        # self.max_iteration_count = 1000

    def run(self, session, coord):
        start_time = timer()
        event_timer = EventTimer()

        many_runs_timeline = TimeLiner()
        def process_trace(run_metadata, many_runs_timeline):
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            many_runs_timeline.update_timeline(chrome_trace)

        # for t in range(self.max_iteration_count):
        for t in range(100000):
            # should_trace = t > 0 and (t % 1 == 0)
            should_trace = False
            should_profile = t > 0 and (t % 100 == 0)

            if should_trace:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                options=None
                run_metadata=None

            if should_profile: event_timer.start()

            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            priorities, obses_t, actions, rewards, obses_tp1, dones = self.dequeue(
                    session=session, options=options, run_metadata=run_metadata)
            if should_profile: event_timer.measure('dequeue')
            if should_trace: process_trace(run_metadata, many_runs_timeline)

            td_error = self.train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards),
                    session=session, options=options, run_metadata=run_metadata)
            if should_profile: event_timer.measure('train')
            if should_trace: process_trace(run_metadata, many_runs_timeline)

            if np.random.random() > 1.0 / 64.0:
                priorities = [np.random.randint(1000000) for _ in range(len(priorities))]
                self.enqueue(priorities, obses_t, actions, rewards, obses_tp1, dones,
                        session=session, options=options, run_metadata=run_metadata)
                if should_profile: event_timer.measure('enqueue')
                if should_trace: process_trace(run_metadata, many_runs_timeline)

            # Update target network periodically.
            if t % self.config.target_update_frequency == 0:
                self.update_target(session=session, options=options, run_metadata=run_metadata)
                if should_profile: event_timer.measure('update_target')
                if should_trace: process_trace(run_metadata, many_runs_timeline)

            event_timer.stop()

            if t % 3000 == 0:
                print(td_error)

            if t % 3000 == 0:
                self.logger.record_tabular("steps", t)
                self.logger.record_tabular("time elapsed", timer() - start_time)
                self.logger.record_tabular("steps/s", t / (timer() - start_time))
                self.logger.record_tabular("queue_size", self.queue_size(session=session))
                self.logger.record_tabular("batch_mean_td_error", np.mean(td_error))
                self.logger.record_tabular("batch_max_td_error", np.max(td_error))
                event_timer.print_shares(self.logger)
                event_timer.print_averages(self.logger)
                self.logger.dump_tabular()

            # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            # chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # many_runs_timeline.update_timeline(chrome_trace)

        many_runs_timeline.save('timeline_merged.json')



class StupidWorker(object):
    def __init__(self, is_chief, env, model):
        self.env = env
        self.is_chief = is_chief

    def run(self, session, coord):
        _ = self.env.reset()

        start_time = timer()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            # action = self.act(obs[None], update_eps=self.exploration.value(t), session=session)[0]
            action = 0
            _, _, done, _ = self.env.step(action)
            if done:
                obs = self.env.reset()

            if self.is_chief and t % 100 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("time elapsed", timer() - start_time)
                logger.record_tabular("steps/s", t / (timer() - start_time))
                logger.dump_tabular()


