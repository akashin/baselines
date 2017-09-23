#!/usr/bin/env python3

import logging

from plumbum import local
from plumbum import BG
from plumbum import cli
from plumbum.cmd import kill, python3

import sys

import atexit

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread_count", type=int, default=1)
    parser.add_argument("--process_count", type=int, default=1)
    parser.add_argument("--env", type=str, default='Pong-v0')
    args = parser.parse_args()

    processes = []
    def cleanup_processes():
        print('Cleaning up processes')
        for process in processes:
            print("Killing {}".format(process.proc.pid))
            kill["-9", process.proc.pid].run()

    atexit.register(cleanup_processes)

    local.env["CUDA_VISIBLE_DEVICES"] = ""

    for i in range(args.process_count):
        print("Starting process {}".format(i))
        f = python3["-m", "baselines.multi_deepq.experiments.custom_cartpole",
                "--worker_count={}".format(args.thread_count),
                "--env_name={}".format(args.env)] & BG(stdout=sys.stdout)
        #taskset -cp $i $pid
        processes.append(f)

    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
