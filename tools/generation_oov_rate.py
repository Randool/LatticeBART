import argparse
import re

reg_patterns = [
    (r"\(.*?\)|\[.*?]", " "),
    (r"\s+", " "),
    ("don't", "do not"),
    ("i'm", "i am"),
    ("i've", "i have"),
    ("you're", "you are"),
    ("we're", "we are"),
    ("they're", "they are"),
    ("that's", "that is"),
    ("it's", "it is"),
    (r"(^\s|\s$)", ""),
]


def parse_key_based_file(file: str):
    data = {}
    with open(file, encoding="utf8") as f:
        for line in f:
            lattice_id, *tokens = line.split()
            data[lattice_id] = " ".join(tokens)
    return data


def process_sent(data: dict):
    for key, val in data.items():  # type: str, str
        val = val.lower()
        for p1, p2 in reg_patterns:
            val = re.sub(p1, p2, val)
        data[key] = val
    return data


def calculate_oov_rate(src: dict, ref: dict):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lat", type=str, required=True, help="lattice file")
    parser.add_argument("-ref", type=str, required=True, help="Model output file")
    args = parser.parse_args()

    src = process_sent(parse_key_based_file(args.i))
    ref = process_sent(parse_key_based_file(args.r))


if __name__ == "__main__":
    main()
