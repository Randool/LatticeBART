"""
训练数据过多，导致需要频繁从文件中读取信息，因此需要预处理lattice文件，记录每个lattice id在不同文件中的位置
"""
import argparse
import glob
import logging
from os.path import realpath


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_pattern", type=str, required=True, help="Files pattern")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output archive file")
    parser.add_argument("-m", "--min_nodes", type=int, default=1, help="Minimum value of the number of nodes")
    parser.add_argument("-M", "--max_nodes", type=int, default=256, help="Maximum value of the number of nodes")
    args = parser.parse_args()
    return args


def read_file(filename: str, id_map: dict, min_nodes: int = 1, max_nodes: int = 256):
    tell = 0
    line_cnt = 0
    lattice_id = None

    with open(filename, encoding="utf8") as f:
        line = f.readline()
        while line:
            if len(line.split()) == 1 and lattice_id is None:
                lattice_id = line.strip()
                tell = f.tell() - len(line)
            elif line.strip() == "":
                if lattice_id is not None:
                    if min_nodes <= line_cnt <= max_nodes:
                        delta = f.tell() - tell
                        if lattice_id in id_map:
                            id_map[lattice_id].append(f"{filename}:{tell}:{delta}")
                        else:
                            id_map[lattice_id] = [f"{filename}:{tell}:{delta}"]
                    line_cnt = 0
                    lattice_id = None
            else:
                line_cnt += 1

            line = f.readline()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S', format='[%(asctime)s] %(levelname)s - %(message)s')

    id_map = {}

    files = sorted(glob.glob(args.file_pattern))
    logging.info(f"{len(files)} files found.")

    for i, file in enumerate(files, start=1):
        logging.info(f"[{i} / {len(files)}] {file}")
        file = realpath(file)
        read_file(file, id_map, args.min_nodes, args.max_nodes)

    with open(args.output, "w", encoding="utf8") as f:
        for key, items in id_map.items():
            line = " ".join([key] + items) + "\n"
            f.write(line)


if __name__ == '__main__':
    main()
