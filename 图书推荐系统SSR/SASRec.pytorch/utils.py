import os
import sys
import copy
import csv
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import torch


# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for _ in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for _ in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(
                    User,
                    usernum,
                    itemnum,
                    batch_size,
                    maxlen,
                    self.result_queue,
                    np.random.randint(2e9)
                ))
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def _resolve_dataset_path(fname):
    """Return an absolute path for the provided dataset argument."""

    if not fname:
        raise ValueError('Dataset name or path must be provided')

    expanded = os.path.expanduser(fname)
    if os.path.isabs(expanded):
        return expanded

    if os.path.exists(expanded):
        return os.path.abspath(expanded)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, expanded),
        os.path.join(base_dir, 'data', f'{expanded}.txt'),
        os.path.join(os.path.dirname(base_dir), expanded),
        os.path.join(os.path.dirname(os.path.dirname(base_dir)), expanded),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    default_txt = os.path.join(base_dir, 'data', f'{fname}.txt')
    if os.path.exists(default_txt):
        return os.path.abspath(default_txt)

    raise FileNotFoundError(f'Cannot locate dataset "{fname}"')


def _split_user_sequences(user_history):
    user_train = {}
    user_valid = {}
    user_test = {}
    for user, items in user_history.items():
        nfeedback = len(items)
        if nfeedback < 3:
            user_train[user] = list(items)
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = list(items[:-2])
            user_valid[user] = [items[-2]]
            user_test[user] = [items[-1]]
    return user_train, user_valid, user_test


def _load_text_dataset(path):
    usernum = 0
    itemnum = 0
    user_history = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            user_history[u].append(i)

    for user in range(1, usernum + 1):
        user_history.setdefault(user, [])

    user_train, user_valid, user_test = _split_user_sequences(user_history)

    id2user = {u: u for u in range(1, usernum + 1)}
    id2item = {i: i for i in range(1, itemnum + 1)}

    return {
        'user_train': user_train,
        'user_valid': user_valid,
        'user_test': user_test,
        'usernum': usernum,
        'itemnum': itemnum,
        'user2id': {u: u for u in range(1, usernum + 1)},
        'item2id': {i: i for i in range(1, itemnum + 1)},
        'id2user': id2user,
        'id2item': id2item,
    }


def _load_csv_dataset(path):
    interactions = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'user_id' not in row or 'book_id' not in row:
                continue
            try:
                user_raw = int(row['user_id'])
                item_raw = int(row['book_id'])
            except ValueError:
                continue
            timestamp = row.get('借阅时间') or row.get('borrow_time') or row.get('timestamp') or ''
            interactions.append((user_raw, item_raw, timestamp))

    if not interactions:
        raise ValueError(f'No valid interactions were found in "{path}"')

    interactions.sort(key=lambda x: (x[0], x[2], x[1]))

    user2id = {}
    item2id = {}
    id2user = {}
    id2item = {}
    user_history = defaultdict(list)

    for user_raw, item_raw, _ in interactions:
        if user_raw not in user2id:
            new_uid = len(user2id) + 1
            user2id[user_raw] = new_uid
            id2user[new_uid] = user_raw
        if item_raw not in item2id:
            new_iid = len(item2id) + 1
            item2id[item_raw] = new_iid
            id2item[new_iid] = item_raw

        user_history[user2id[user_raw]].append(item2id[item_raw])

    usernum = len(user2id)
    itemnum = len(item2id)

    for user in range(1, usernum + 1):
        user_history.setdefault(user, [])

    user_train, user_valid, user_test = _split_user_sequences(user_history)

    return {
        'user_train': user_train,
        'user_valid': user_valid,
        'user_test': user_test,
        'usernum': usernum,
        'itemnum': itemnum,
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item,
    }


# train/val/test data generation
def data_partition(fname):
    path = _resolve_dataset_path(fname)

    if os.path.isdir(path):
        candidate = os.path.join(path, 'inter_reevaluation.csv')
        if os.path.isfile(candidate):
            return _load_csv_dataset(candidate)
        raise FileNotFoundError(f'No interaction csv file found inside directory "{path}"')

    _, ext = os.path.splitext(path)
    if ext.lower() == '.csv':
        return _load_csv_dataset(path)

    return _load_text_dataset(path)


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def _prepare_topks(topks):
    if topks is None:
        return [10]
    unique = sorted({int(k) for k in topks if int(k) > 0})
    if not unique:
        raise ValueError('At least one positive integer top-k value must be provided')
    return unique


def _init_metric_buckets(topks):
    return {k: {'NDCG': 0.0, 'HR': 0.0} for k in topks}


def _finalize_metrics(buckets, valid_user):
    if valid_user == 0:
        return {k: (0.0, 0.0) for k in buckets}
    return {k: (vals['NDCG'] / valid_user, vals['HR'] / valid_user) for k, vals in buckets.items()}


def evaluate(model, dataset, args, topks=None):
    data = copy.deepcopy(dataset)
    train = data['user_train']
    valid = data['user_valid']
    test = data['user_test']
    usernum = data['usernum']
    itemnum = data['itemnum']

    valid_user = 0.0
    topk_values = _prepare_topks(topks)
    metric_buckets = _init_metric_buckets(topk_values)

    users = random.sample(range(1, usernum + 1), 10000) if usernum > 10000 else range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]  # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        for k in topk_values:
            if rank < k:
                metric_buckets[k]['NDCG'] += 1 / np.log2(rank + 2)
                metric_buckets[k]['HR'] += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return _finalize_metrics(metric_buckets, valid_user)


# evaluate on val set
def evaluate_valid(model, dataset, args, topks=None):
    data = copy.deepcopy(dataset)
    train = data['user_train']
    valid = data['user_valid']
    usernum = data['usernum']
    itemnum = data['itemnum']

    valid_user = 0.0
    topk_values = _prepare_topks(topks)
    metric_buckets = _init_metric_buckets(topk_values)
    users = random.sample(range(1, usernum + 1), 10000) if usernum > 10000 else range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        for k in topk_values:
            if rank < k:
                metric_buckets[k]['NDCG'] += 1 / np.log2(rank + 2)
                metric_buckets[k]['HR'] += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return _finalize_metrics(metric_buckets, valid_user)


def generate_topk_recommendations(model, dataset, args, topks, output_path):
    if not topks:
        raise ValueError('At least one top-k value must be provided')

    unique_topks = sorted(set(int(k) for k in topks if k > 0))
    if not unique_topks:
        raise ValueError('Top-k values must be positive integers')

    train = dataset['user_train']
    valid = dataset['user_valid']
    test = dataset['user_test']
    usernum = dataset['usernum']
    itemnum = dataset['itemnum']
    id2user = dataset.get('id2user', {})
    id2item = dataset.get('id2item', {})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = ['user_id'] + [f'top{k}' for k in unique_topks]

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        all_items = np.arange(1, itemnum + 1)

        for u in range(1, usernum + 1):
            history = []
            history.extend(train.get(u, []))
            history.extend(valid.get(u, []))
            history.extend(test.get(u, []))

            if not history:
                continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            recent = history[-args.maxlen:]
            seq[-len(recent):] = recent

            seen = set(history)
            candidates = np.setdiff1d(all_items, np.fromiter(seen, dtype=np.int32), assume_unique=False)
            if candidates.size == 0:
                continue

            predictions = model.predict(
                np.array([u]),
                np.array([seq]),
                candidates.tolist()
            )[0]

            if isinstance(predictions,torch.Tensor):
                predictions=predictions.detach().cpu().numpy()
            ranking = np.argsort(-predictions)

            row = {'user_id': id2user.get(u, u)}
            for k in unique_topks:
                top_indices = ranking[:k]
                recommended = [id2item.get(int(candidates[idx]), int(candidates[idx])) for idx in top_indices]
                row[f'top{k}'] = ' '.join(str(item) for item in recommended)

            writer.writerow(row)
