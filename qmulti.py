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

def run_doom(batch_size=32, learning_rate=1e-4, actor_count=1, tf_thread_count=8, env='ppaquette/BasicDoom-v0'):
    print("Starting doom training")
    f = python3["-m", "baselines.qdqn.experiments.doom",
            "--batch_size={}".format(batch_size),
            "--learning_rate={}".format(learning_rate),
            "--actor_count={}".format(actor_count),
            "--tf_thread_count={}".format(tf_thread_count),
            "--exploration_length={}".format(50000),
            "--env_name={}".format(env)] & BG(stdout=sys.stdout, stderr=sys.stderr)
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
            run_doom(batch_size=batch_size,
                    learning_rate=learning_rate,
                    actor_count=args.actor_count,
                    tf_thread_count=args.tf_thread_count,
                    env=args.env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--actor_count", type=int, default=1)
    parser.add_argument("--tf_thread_count", type=int, default=8)
    parser.add_argument("--env", type=str, default='Pong-v0')
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



if __name__ == "__main__":
    main()
