#!/usr/bin/env python3

import logging

from plumbum import local
from plumbum import BG
from plumbum import cli
from plumbum.cmd import kill, python3
import plumbum.commands.processes

import sys

import atexit

import argparse

processes = []
def cleanup_processes():
    global processes
    print('Cleaning up processes')
    for process in processes:
        print("Killing {}".format(process.proc.pid))
        if not process.ready():
            kill["-9", process.proc.pid].run()

def wait_for(process):
    try:
        process.wait()
    except plumbum.commands.processes.ProcessExecutionError as e:
        print("Process failed with {}".format(e))

def run_doom_qdqn(batch_size=32, learning_rate=1e-4, actor_count=1, tf_thread_count=8, 
        target_update_frequency=500, num_iterations=None, env='ppaquette/BasicDoom-v0'):
    if num_iterations is None:
        num_iterations = 2500

    print("Starting doom training with QDQN")
    f = python3["-m", "baselines.qdqn.experiments.doom",
            "--batch_size={}".format(batch_size),
            "--learning_rate={}".format(learning_rate),
            "--actor_count={}".format(actor_count),
            "--tf_thread_count={}".format(tf_thread_count),
            "--target_update_frequency={}".format(target_update_frequency),
            "--num_iterations={}".format(num_iterations),
            "--cleanup={}".format(True),
            "--env_name={}".format(env)] & BG(stdout=sys.stdout, stderr=sys.stderr)
    #taskset -cp $i $pid
    wait_for(f)

def run_doom_dqn(batch_size=32, learning_rate=1e-4, train_frequency=4,
        worker_count=1, tf_thread_count=8,
        target_update_frequency=500, num_iterations=None, env='ppaquette/BasicDoom-v0'):
    if num_iterations is None:
        num_iterations = 10000

    print("Starting doom training with DQN")
    f = python3["-m", "baselines.multi_deepq.experiments.doom",
            "--batch_size={}".format(batch_size),
            "--train_frequency={}".format(train_frequency),
            "--learning_rate={}".format(learning_rate),
            "--worker_count={}".format(args.thread_count),
            "--target_update_frequency={}".format(target_update_frequency),
            "--num_iterations={}".format(num_iterations),
            "--tf_thread_count={}".format(args.tf_thread_count),
            "--env_name={}".format(args.env)] & BG(stdout=sys.stdout, stderr=sys.stderr)
    #taskset -cp $i $pid
    wait_for(f)

def run_doom_sweep(args, learning_rates=None, batch_sizes=None):
    if not learning_rates:
        learning_rates = [args.learning_rate]

    if not batch_sizes:
        batch_sizes = [args.batch_size]

    print("Running doom sweep with {} parameter(s)".format(len(learning_rates) * len(batch_sizes)))

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            if args.algo == "qdqn":
                run_doom_qdqn(batch_size=batch_size,
                        learning_rate=learning_rate,
                        actor_count=args.actor_count,
                        tf_thread_count=args.tf_thread_count,
                        target_update_frequency=args.target_update_frequency,
                        num_iterations=args.num_iterations,
                        env=args.env)
            elif args.algo == "dqn":
                run_doom_dqn(batch_size=batch_size,
                        learning_rate=learning_rate,
                        tf_thread_count=args.tf_thread_count,
                        target_update_frequency=args.target_update_frequency,
                        num_iterations=args.num_iterations,
                        env=args.env)
            else:
                raise ValueError("Invalid algo: {}".format(args.algo))

def run_atari(batch_size=32, learning_rate=1e-4, actor_count=1, tf_thread_count=8,
        target_update_frequency=1000, env='PongNoFrameskip-v4'):
    print("Starting Atari training")
    f = python3["-m", "baselines.qdqn.experiments.train_atari",
            "--batch_size={}".format(batch_size),
            "--learning_rate={}".format(learning_rate),
            "--actor_count={}".format(actor_count),
            "--tf_thread_count={}".format(tf_thread_count),
            "--target_update_frequency={}".format(target_update_frequency),
            "--num_iterations={}".format(int(5e7)),
            "--env_name={}".format(env)] & BG(stdout=sys.stdout, stderr=sys.stderr)
    #taskset -cp $i $pid
    wait_for(f)

def run_atari_sweep(args, learning_rates=None, batch_sizes=None):
    if not learning_rates:
        learning_rates = [args.learning_rate]

    if not batch_sizes:
        batch_sizes = [args.batch_size]

    print("Running Atari sweep with {} parameter(s)".format(len(learning_rates) * len(batch_sizes)))

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            run_atari(batch_size=batch_size,
                    learning_rate=learning_rate,
                    actor_count=args.actor_count,
                    tf_thread_count=args.tf_thread_count,
                    target_update_frequency=args.target_update_frequency,
                    env=args.env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--actor_count", type=int, default=1)
    parser.add_argument("--tf_thread_count", type=int, default=8)
    parser.add_argument("--target_update_frequency", type=int, default=500)
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--algo", type=str, default="qdqn")
    parser.add_argument("--env", type=str, default="Pong-v0")
    args = parser.parse_args()

    atexit.register(cleanup_processes)

    # Don't use GPU.
    # local.env["CUDA_VISIBLE_DEVICES"] = ""

    if 'ppaquette' in args.env:
        run_doom_sweep(args)

        # run_doom_sweep(
                # args,
                # batch_sizes=[32, 64, 128, 256],
                # learning_rates=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    else:
        run_atari_sweep(args)




if __name__ == "__main__":
    main()
