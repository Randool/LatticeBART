import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=str, required=True)
    parser.add_argument("--score_file", type=str, required=True)
    parser.add_argument("--lm_scale", type=float, default=5)
    parser.add_argument("--write", "-o", type=str, required=True)
    args = parser.parse_args()

    sents = {}
    scores = {}

    with open(args.text_file, encoding='utf8') as f:
        for line in f:
            if len(line) < 1 or line.isspace():
                continue
            key, sent = line.strip().split(maxsplit=1)
            sents[key] = sent

    with open(args.score_file, encoding='utf8') as f:
        for line in f:
            if len(line) < 1 or line.isspace():
                continue
            key, ac, lm = line.split()
            scores[key] = float(ac) - float(lm) * args.lm_scale

    buckets = {}
    for key in sents:
        prefix = "-".join(key.split("-")[:-1])
        if prefix not in buckets:
            buckets[prefix] = [(scores[key], sents[key])]
        else:
            buckets[prefix].append((scores[key], sents[key]))

    results = []
    for key in sorted(buckets.keys()):
        bucket = sorted(buckets[key], key=lambda x: x[0], reverse=True)
        results.append(f"{key} {bucket[0][1]}")

    with open(args.write, "w", encoding="utf8") as f:
        f.write("\n".join(results) + '\n')


if __name__ == '__main__':
    main()
