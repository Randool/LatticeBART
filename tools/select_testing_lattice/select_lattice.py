import random


def main():
    pairs = []
    with open("/mnt/lustre/sjtu/home/lfd98/Dataset/mr_fisher_swbd/test.txt") as f:
        for line in f:
            if len(line) <= 2:
                continue
            try:
                key, sent = line.strip().split(maxsplit=1)
            except ValueError:
                continue
            pairs.append((key, sent))

    len_threshold_h = [5, 8, 16, 24, 32]
    len_threshold_H = [h + 3 for h in len_threshold_h]
    len_threshold_H[-1] = 1000
    buckets = [[] for _ in range(len(len_threshold_H))]

    N = 8
    batch = 3

    for pair in pairs:
        key, sent = pair
        _len = len(sent.split())
        for i in range(len(len_threshold_H)):
            if len_threshold_h[i] <= _len <= len_threshold_H[i]:
                buckets[i].append(pair)
                break

    for i, bucket in enumerate(buckets):
        if len(bucket) < N * batch:
            print(f"The number of items in bucket {i} ({len(bucket)}) is not enough.")
            print(list(map(len, buckets)))
            return
        random.shuffle(bucket)

    for i in range(batch):
        with open(f"batch_{i}.txt", "w", encoding="utf8") as f:
            for j in range(len(len_threshold_h)):
                if j == 0:
                    continue
                pairs = buckets[j][i * N: (i + 1) * N]
                for key, sent in pairs:
                    f.write(f"{key}\t{sent}\n")


if __name__ == "__main__":
    main()
