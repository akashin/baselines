import tensorflow as tf
import tensorflow.contrib.layers as layers
import itertools
import numpy as np

from tensorflow.python.client import timeline

from tensorflow.contrib.staging import StagingArea

from timeit import default_timer as timer

import baselines.common.tf_util as U

from baselines import logger

from baselines import qdqn
from baselines.qdqn.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule

from collections import namedtuple, defaultdict

import json
import os
import time

from baselines.common.misc_util import (
        pickle_load,
        relatively_safe_pickle_dump
        )


class Config(object):

    def __init__(self):
        self.batch_size = 64
        self.gamma = 0.99
        self.actor_count = 1
        # self.num_iterations = 50000
        self.num_iterations = 2e7
        self.exploration_schedule = "linear"
        self.learning_rate = 5e-3
        self.tf_thread_count = 8
        self.target_update_frequency = 200
        self.params_update_frequency = 1000
        self.queue_capacity = 2 ** 17
        self.replay_buffer_size = int(1e6 / 20)

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
        self.global_step = tf.train.get_global_step()
        self.should_render = should_render
        self.logger = logger

        with tf.device('/cpu:0'):
            self.act, self.update_params, self.debug = qdqn.build_act(
                    make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
                    q_func=model,
                    num_actions=env.action_space.n,
                    scope="actor_{}".format(index),
                    learner_scope="learner",
                    reuse=False)

            obs_t_input = tf.placeholder(tf.int8, env.observation_space.shape, name="obs_t")
            act_t_ph = tf.placeholder(tf.int32, env.action_space.shape, name="action")
            rew_t_ph = tf.placeholder(tf.float32, [], name="reward")
            obs_tp1_input = tf.placeholder(tf.int8, env.observation_space.shape, name="obs_tp1")
            done_mask_ph = tf.placeholder(tf.float32, [], name="done")
            global_step_ph = tf.placeholder(tf.int32, [], name="sample_global_step")
            enqueue_op = queue.enqueue(
                    [obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph, global_step_ph])
            self.enqueue = U.function(
                    [obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph, global_step_ph], enqueue_op)

        self.max_iteration_count = self.config.num_iterations

        if self.config.exploration_schedule == "linear":
            # Create the schedule for exploration starting from 1 (every action is random) down to
            # 0.02 (98% of actions are selected according to values predicted by the model).
            self.exploration = LinearSchedule(
                    schedule_timesteps=self.config.num_iterations / 4, initial_p=1.0, final_p=0.02)
        elif self.config.exploration_schedule == "piecewise":
            approximate_num_iters = self.config.num_iterations
            self.exploration = PiecewiseSchedule([
                (0, 1.0),
                (approximate_num_iters / 50, 0.1),
                (approximate_num_iters / 5, 0.01)
            ], outside_value=0.01)
        else:
            raise ValueError("Bad exploration schedule")

    def run(self, session, coord):
        episode_rewards = [0.0]
        obs = self.env.reset()
        done = False

        global_step = session.run(self.global_step)
        exploration_value = self.exploration.value(global_step)
        print("Starting acting from step {}".format(global_step))

        start_time = timer()
        event_timer = EventTimer()
        for t in itertools.count():
            if coord.should_stop():
                break

            if t % 100 == 0:
                global_step = session.run(self.global_step)
                exploration_value = self.exploration.value(global_step)

            if t > 0 and t % 10 == 0:
                event_timer.start()
            # Take action and update exploration to the newest value
            action = self.act(np.array(obs)[None], update_eps=exploration_value, session=session)[0]
            if done and len(episode_rewards) % 10 == 0:
                print(self.debug["q_values"](obs[None], session=session))
                # self.update_params(session=session)
                # print(self.debug["q_values"](obs[None], session=session))

            event_timer.measure('act')
            new_obs, rew, done, _ = self.env.step(action)
            event_timer.measure('step')
            # Store transition in the replay buffer.
            self.enqueue(obs, action, rew, new_obs, float(done), global_step, session=session)
            # self.replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = self.env.reset()
                episode_rewards.append(0)

            event_timer.measure('queue')

            # show_episode = len(episode_rewards) > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            show_episode = len(episode_rewards) % 10 == 0
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
                self.logger.record_tabular("global_step", global_step)
                self.logger.record_tabular("episodes", len(episode_rewards))
                self.logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                self.logger.record_tabular("time elapsed", timer() - start_time)
                self.logger.record_tabular("steps/s", t / (timer() - start_time))
                self.logger.record_tabular("% time spent exploring", int(100 * exploration_value))
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


class TFReplayBuffer(object):
    def __init__(self, replay_buffer_size, observation_space, action_space):
        self.capacity = replay_buffer_size

        with tf.variable_scope("replay_buffer"):
            self.current_size = tf.Variable(0, "buffer_size")
            self.size = U.function([], self.current_size)

            batch_shape = []
            self.observations = tf.TensorArray(element_shape=(batch_shape + list(observation_space.shape), dtype==np.float32),
                    name="observations")
            self.actions = tf.TensorArray(element_shape=(batch_shape + list(action_space.shape), dtype==np.int32),
                    name="actions")
            self.rewards = tf.TensorArray(element_shape=(batch_shape), dtype==np.float32, name="rewards")
            self.next_observations = tf.TensorArray(element_shape=(batch_shape + list(observation_space.shape)), dtype==np.float32),
                    name="next_observations")
            self.dones = tf.TensorArray(element_shape=(batch_shape), dtype==np.float32, name="dones")

    def add(self, obs_t, act_t, rew_t, obs_tp1, done):
        ops = tf.group(*[
            tf.assign(self.observations[self.current_size], obs_t),
            tf.assign(self.actions[self.current_size], act_t),
            tf.assign(self.rewards[self.current_size], rew_t),
            tf.assign(self.next_observations[self.current_size], obs_tp1),
            tf.assign(self.dones[self.current_size], done),
        ])
        with tf.control_dependencies([ops]):
            increment_size = tf.assign(self.current_size, (self.current_size + 1) % self.capacity)

        return tf.group(*[ops, increment_size])

    def sample(self, batch_size):
        logits = tf.log(tf.ones([self.current_size], dtype=tf.float32) / tf.cast(self.current_size, tf.float32))
        indices = tf.multinomial([logits], num_samples=batch_size)
        return tf.gather(self.observations, indices)[0], \
                tf.gather(self.actions, indices)[0], \
                tf.gather(self.rewards, indices)[0], \
                tf.gather(self.next_observations, indices)[0], \
                tf.gather(self.dones, indices)[0] \


class Trainer(object):
    def __init__(self, config, actor_queue, learner_queue, observation_space, action_space,
            logger):
        self.actor_queue = actor_queue
        self.learner_queue = learner_queue
        self.logger = logger

        self.batch_size = config.batch_size
        # self.replay_buffer = TFReplayBuffer(config.replay_buffer_size, observation_space,
                # action_space)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        with tf.device('/cpu:0'):
            batch_shape = [self.batch_size]
            obs_t_input = tf.placeholder(tf.int8, batch_shape + list(observation_space.shape), name="obs_t")
            act_t_ph = tf.placeholder(tf.int32, batch_shape + list(action_space.shape), name="action")
            rew_t_ph = tf.placeholder(tf.float32, batch_shape, name="reward")
            obs_tp1_input = tf.placeholder(tf.int8, batch_shape + list(observation_space.shape), name="obs_tp1")
            done_mask_ph = tf.placeholder(tf.float32, batch_shape, name="done")

            # add_to_replay_buffer_op = self.replay_buffer.add(obs_t_input, act_t_ph,
                    # rew_t_ph, obs_tp1_input, done_mask_ph)
            # self.add_to_replay_buffer = U.function([obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph],
                    # add_to_replay_buffer_op)

            dequeue_op = actor_queue.dequeue_up_to(256)
            # obs_t_inputs, actions, rewards, obs_tp1_inputs, dones, global_steps = dequeue_op
            self.dequeue = U.function([], dequeue_op)

            enqueue_batch_op = learner_queue.put(
                    [obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph])

            self.enqueue = U.function([obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph],
                    enqueue_batch_op)

            actor_queue_size_op = actor_queue.size()
            self.actor_queue_size = U.function([], actor_queue_size_op)

            learner_queue_size_op = learner_queue.size()
            self.learner_queue_size = U.function([], learner_queue_size_op)

            # enqueue_op = learner_queue.put(self.replay_buffer.sample(self.batch_size))
            # self.enqueue = U.function([], enqueue_op)

            # enqueue_ops = []
            # for i in range(40):
                # enqueue_op = learner_queue.put(
                        # [obs_t_inputs, actions, rewards, obs_tp1_inputs, dones])
                # enqueue_ops.append(enqueue_op)
            # self.enqueue = U.function([], tf.group(*enqueue_ops))

    def run(self, session, coord):
        t = 0
        total_sampled = 0
        last_time = timer()
        # sample_count = 100
        target_occupancy = 15000

        event_timer = EventTimer()
        while not coord.should_stop():
            should_profile = t > 0 and t % 10 == 0
            if should_profile: event_timer.start()

            t += 1
            obs_t_inputs, actions, rewards, obs_tp1_inputs, dones, global_steps = self.dequeue(session=session)
            for i in range(len(rewards)):
                # self.add_to_replay_buffer(obs_t_inputs[i], actions[i], rewards[i], obs_tp1_inputs[i], dones[i], session=session)
                self.replay_buffer.add(obs_t_inputs[i], actions[i], rewards[i], obs_tp1_inputs[i], dones[i])
            if should_profile: event_timer.measure('dequeue')

            if t > 0 and t % 10 == 0:
                current_time = timer()
                sample_rate = total_sampled / (current_time - last_time)
                print("Sample rate: {}".format(sample_rate))
                last_time = current_time
                total_sampled = 0

                # if sample_rate < target_rate * 0.9:
                    # sample_count *= 1.1
                # elif sample_rate > target_rate * 1.1:
                    # sample_count /= 1.1

                # print("New sample count: {}".format(sample_count))

            # sample_now = min(int(sample_size), len(self.replay_buffer))
            # if self.replay_buffer.size(session=session) >= 1000:
            if len(self.replay_buffer) >= 1000:
                # queue_size = self.learner_queue_size(session=session)
                # sample_now = min(max(0, target_occupancy - queue_size), 8096)
                sample_now = 10000
                for i in range(sample_now // self.batch_size + 1):
                    # omg = self.replay_buffer.sample(self.batch_size)
                    self.enqueue(*self.replay_buffer.sample(self.batch_size), session=session)
                    # self.enqueue(session=session)
                    total_sampled += self.batch_size

                if should_profile: event_timer.measure('enqueue')

            event_timer.stop()
            if should_profile:
                self.logger.record_tabular("actor_queue_size", self.actor_queue_size(session=session))
                event_timer.print_shares(self.logger)
                event_timer.print_averages(self.logger)
                self.logger.dump_tabular()

class Learner(object):
    def __init__(self, save_path, observation_space, action_space, model, queue, config, logger):
        self.config = config
        self.queue = queue
        self.logger = logger
        self.save_path = save_path
        self.batch_size = config.batch_size

        queue_size_op = self.queue.size()
        self.queue_size = U.function([], queue_size_op)

        with tf.device('/cpu:0'):
            self.global_step = tf.train.create_global_step()
            self.update_global_step = tf.assign_add(self.global_step, 1)

        with tf.device('/gpu:0'):
            dequeue_op = self.queue.get()
            self.act, self.train, self.update_target, self.debug = qdqn.build_train(
                    make_obs_ph=lambda name: U.Uint8Input(observation_space.shape, name=name),
                    q_func=model,
                    num_actions=action_space.n,
                    gamma=config.gamma,
                    optimizer=tf.train.AdamOptimizer(learning_rate=config.learning_rate, epsilon=1e-4),
                    train_inputs=dequeue_op,
                    scope="learner",
                    grad_norm_clipping=10,
                    reuse=False)

        self.num_iters = 0
        self.max_iteration_count = self.config.num_iterations

        self.checkpoint_frequency = max(self.max_iteration_count / 1000, 10000)

        self.log_frequency = 500

    def load(self, path, session):
        """Load model if present at the specified path."""
        if path is None:
            return

        state_path = os.path.join(os.path.join(path, 'training_state.pkl.zip'))
        found_model = os.path.exists(state_path)
        if found_model:
            state = pickle_load(state_path, compression=True)
            model_dir = "model-{}".format(state["num_iters"])
            U.load_state(os.path.join(path, model_dir, "saved"), session=session)
            self.logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))

            if state is not None:
                self.num_iters = state["num_iters"]
                # self.replay_buffer = state["replay_buffer"],
                # monitored_env.set_state(state["monitor_state"])

    def save(self, path, session):
        """This function checkpoints the model and state of the training algorithm."""
        if path is None:
            return

        state = {
            # 'replay_buffer': self.replay_buffer,
            'num_iters': self.num_iters,
            # 'monitor_state': monitored_env.get_state(),
        }

        start_time = time.time()
        model_dir = "model-{}".format(state["num_iters"])
        U.save_state(os.path.join(path, model_dir, "saved"), session=session)
        relatively_safe_pickle_dump(state, os.path.join(path, 'training_state.pkl.zip'), compression=True)
        # relatively_safe_pickle_dump(state["monitor_state"], os.path.join(path, 'monitor_state.pkl'))
        logger.log("Saved model in {} seconds\n".format(time.time() - start_time))

    def run(self, session, coord):
        start_time = timer()
        event_timer = EventTimer()

        many_runs_timeline = TimeLiner()
        def process_trace(run_metadata, many_runs_timeline):
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            many_runs_timeline.update_timeline(chrome_trace)

        start_step = session.run(self.global_step)
        print("Starting training from step {}".format(start_step))
        for t in range(start_step, self.max_iteration_count):
            # should_trace = t > 0 and (t % 1 == 0)
            should_trace = False
            should_profile = t > 500 and (t % self.log_frequency == 0)
            if should_trace:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                options=None
                run_metadata=None

            if t != start_step and t % self.checkpoint_frequency == 0:
                self.num_iters = t
                self.save(self.save_path, session)

            if should_profile: event_timer.start()

            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            td_error = self.train(np.ones(self.batch_size),
                    session=session, options=options, run_metadata=run_metadata)
            if should_profile: event_timer.measure('train')
            if should_trace: process_trace(run_metadata, many_runs_timeline)

            # Update target network periodically.
            if t % self.config.target_update_frequency == 0:
                self.update_target(session=session, options=options, run_metadata=run_metadata)
                if should_profile: event_timer.measure('update_target')
                if should_trace: process_trace(run_metadata, many_runs_timeline)

            event_timer.stop()

            if t > 0 and t % self.log_frequency == 0:
                steps_per_second = (t - start_step) / (timer() - start_time)
                frames_per_second = self.batch_size * steps_per_second
                self.logger.record_tabular("step", t)
                self.logger.record_tabular("time elapsed", timer() - start_time)
                self.logger.record_tabular("steps/s", steps_per_second)
                self.logger.record_tabular("frames/s", frames_per_second)
                self.logger.record_tabular("queue_size", self.queue_size(session=session))
                self.logger.record_tabular("batch_mean_td_error", np.mean(td_error))
                self.logger.record_tabular("batch_max_td_error", np.max(td_error))
                # self.logger.record_tabular("td_error", str(td_error))
                event_timer.print_shares(self.logger)
                event_timer.print_averages(self.logger)
                self.logger.dump_tabular()

            session.run(self.update_global_step)

        self.num_iters = self.max_iteration_count
        self.save(self.save_path, session)
        many_runs_timeline.save('timeline_merged.json')
        coord.request_stop()



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


