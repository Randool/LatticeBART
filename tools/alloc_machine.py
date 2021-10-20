import argparse
import os
from pathlib import Path

SLURM_HEAD = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:{gres}
#SBATCH --mem={mem}G
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --exclude={exclude}
#SBATCH --array=0

"""

SLURM_CMD = """
echo ${{CUDA_VISIBLE_DEVICES}}

while true; do
    sleep 100
done
"""


def main():
    home = str(Path.home())

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--job-name", type=str, help="job-name")
    parser.add_argument("-p", "--partition", type=str, default="2080ti,gpu", help="partition")
    parser.add_argument("-c", "--cpus-per-task", type=int, default=12, help="cpus-per-task")
    parser.add_argument("-g", "--gres", type=int, default=4, help="cpus-per-task")
    parser.add_argument("-m", "--mem", type=int, default=60, help="mem (G)")
    parser.add_argument("-o", "--output", type=str, default=f"{home}/tmp/debug-%A-%a.out", help="output")
    parser.add_argument("-e", "--error", type=str, default=f"{home}/tmp/debug-%A-%a.err", help="error")
    parser.add_argument("-x", "--exclude", type=str,
                        default="gqxx-01-[003,006,011,016,020,033,083,093,105,110,125,139,145,146,152,167,174]")
    args = parser.parse_args()
    args.partition = "cpu" if args.gres == 0 else args.partition

    if args.job_name is None:
        # Auto-config job name
        args.job_name = f"cpu{args.cpus_per_task}_gpu{args.gres}_mem{args.mem}"

    filename = f"{home}/tmp/alloc_{args.job_name}.sbatch"
    print(f"tmpfile: {filename}")

    with open(filename, "w", encoding="utf8") as f:
        content = SLURM_HEAD.format(**vars(args)) + SLURM_CMD
        f.write(content)

    print(os.system(f"sbatch {filename}"))


if __name__ == "__main__":
    main()
