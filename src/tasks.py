"""Few-shot episode generators for 4 tasks. All synthetic / sklearn-builtin.

Splits are CLASS-DISJOINT where the class count allows it (vision, coding)
and exemplar-disjoint otherwise (tabular, chess). Every generator is seeded.
"""
import numpy as np

TASK_INFO = {
    # n_way used for both train and test; input_dim fixed per task
    "vision":  {"n_way": 5, "input_dim": 64},
    "tabular": {"n_way": 3, "input_dim": 13},
    "chess":   {"n_way": 2, "input_dim": 18},
    "coding":  {"n_way": 4, "input_dim": 22},
}

_CACHE = {}


def _digits_pseudo_classes(seed=0):
    from sklearn.datasets import load_digits
    key = ("digits", seed)
    if key in _CACHE:
        return _CACHE[key]
    d = load_digits()
    X = d.data / 16.0  # (1797, 64)
    y = d.target
    rng = np.random.RandomState(seed)
    # 10 digits x 2 rotations = 20 pseudo-classes
    pools = {}
    for digit in range(10):
        base = X[y == digit]
        for r, tag in enumerate([0, 1]):
            if r == 1:
                rot = np.stack([np.rot90(v.reshape(8, 8), k=1).ravel() for v in base])
            else:
                rot = base
            pools[(digit, tag)] = rot[rng.permutation(len(rot))]
    _CACHE[key] = pools
    return pools


def _tabular_pools(seed=0):
    from sklearn.datasets import load_wine, load_iris
    from sklearn.preprocessing import StandardScaler
    key = ("tab", seed)
    if key in _CACHE:
        return _CACHE[key]
    w = load_wine(); ir = load_iris()
    Xw = StandardScaler().fit_transform(w.data)
    Xi = np.zeros((len(ir.data), 13)); Xi[:, :4] = StandardScaler().fit_transform(ir.data)
    pools = {}
    for c in range(3):
        pools[("wine", c)] = Xw[w.target == c]
        pools[("iris", c)] = Xi[ir.target == c]
    _CACHE[key] = pools
    return pools


def _chess_dataset(seed=0, n=1200):
    key = ("chess", seed, n)
    if key in _CACHE:
        return _CACHE[key]
    import chess
    rng = np.random.RandomState(seed)
    PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9}

    def feats(board: chess.Board):
        # 20-d handcrafted feature vector
        white_mat = sum(len(board.pieces(pt, chess.WHITE)) * v for pt, v in PIECE_VAL.items())
        black_mat = sum(len(board.pieces(pt, chess.BLACK)) * v for pt, v in PIECE_VAL.items())
        mine = board.turn
        mob = len(list(board.legal_moves))
        check = float(board.is_check())
        # piece counts by type for side to move (5) + opp (5)
        mine_c = [len(board.pieces(pt, mine)) for pt in
                  [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]]
        opp_c = [len(board.pieces(pt, not mine)) for pt in
                 [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]]
        # attackers on king square
        ksq = board.king(mine)
        n_att = len(list(board.attackers(not mine, ksq))) if ksq is not None else 0
        v = np.array([white_mat - black_mat, white_mat, black_mat, mob / 40.0, check] +
                     [c / 8.0 for c in mine_c] + [c / 8.0 for c in opp_c] +
                     [n_att / 4.0, float(board.turn), board.fullmove_number / 40.0],
                     dtype=np.float32)
        return v  # 5+5+5+... = 20

    def has_mate_in_1(board: chess.Board):
        for mv in board.legal_moves:
            board.push(mv)
            m = board.is_checkmate()
            board.pop()
            if m:
                return True
        return False

    X, y = [], []
    need = n // 2
    got = {0: 0, 1: 0}
    guard = 0
    while (got[0] < need or got[1] < need) and guard < 60000:
        guard += 1
        b = chess.Board()
        for _ in range(int(rng.randint(8, 28))):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(moves[rng.randint(len(moves))])
            if b.is_game_over():
                break
        if b.is_game_over():
            continue
        label = 1 if has_mate_in_1(b) else 0
        if got[label] >= need:
            continue
        got[label] += 1
        X.append(feats(b)); y.append(label)
    X = np.stack(X).astype(np.float32); y = np.array(y)
    _CACHE[key] = (X, y)
    return X, y


# ---- coding DSL ----
def _apply_transform(tid, lst):
    l = list(lst)
    if tid == 0:   # reverse
        return l[::-1]
    if tid == 1:   # sort asc
        return sorted(l)
    if tid == 2:   # sort desc
        return sorted(l, reverse=True)
    if tid == 3:   # +1 mod 10
        return [(v + 1) % 10 for v in l]
    if tid == 4:   # +2 mod 10
        return [(v + 2) % 10 for v in l]
    if tid == 5:   # x2 mod 10
        return [(v * 2) % 10 for v in l]
    if tid == 6:   # rotate left 1
        return l[1:] + l[:1]
    if tid == 7:   # cumsum mod 10
        s, out = 0, []
        for v in l:
            s = (s + v) % 10; out.append(s)
        return out
    if tid == 8:   # filter even, pad with 0
        ev = [v for v in l if v % 2 == 0]
        return (ev + [0] * len(l))[:len(l)]
    if tid == 9:   # mod10 of x+5
        return [(v + 5) % 10 for v in l]
    raise ValueError(tid)


def _enc_list(lst, L=10):
    v = np.zeros(L + 1, dtype=np.float32)
    v[0] = len(lst) / 10.0
    for i, x in enumerate(lst[:L]):
        v[1 + i] = x / 10.0
    return v  # 11-d


def _coding_pairs(seed=0, per=120, L=10):
    key = ("code", seed, per)
    if key in _CACHE:
        return _CACHE[key]
    rng = np.random.RandomState(seed)
    pools = {}
    for tid in range(10):
        rows = []
        for _ in range(per):
            ln = int(rng.randint(6, L + 1))
            inp = [int(rng.randint(0, 10)) for _ in range(ln)]
            out = _apply_transform(tid, inp)
            rows.append(np.concatenate([_enc_list(inp), _enc_list(out)]).astype(np.float32))
        pools[tid] = np.stack(rows)
    _CACHE[key] = pools
    return pools


def sample_episode(task, split, n_way, k_shot, q_query, rng):
    """Returns (sx, sy, qx, qy) as float32/int64 numpy arrays.

    split: 'train' or 'test'. Classes for train vs test are disjoint for
    vision and coding; exemplar-disjoint otherwise.
    """
    assert split in ("train", "test")
    if task == "vision":
        pools = _digits_pseudo_classes(seed=0)
        keys = sorted(pools)
        train_keys = [k for k in keys if not (k[0] >= 7 and k[1] == 1)][:12]
        test_keys = [k for k in keys if k not in train_keys][:8]
        pool = train_keys if split == "train" else test_keys
        chosen = [pool[i] for i in rng.choice(len(pool), n_way, replace=False)]
        sx, sy, qx, qy = [], [], [], []
        for li, k in enumerate(chosen):
            arr = pools[k]
            idx = rng.choice(len(arr), k_shot + q_query, replace=False)
            samp = arr[idx]
            sx.append(samp[:k_shot]); sy += [li] * k_shot
            qx.append(samp[k_shot:]); qy += [li] * q_query
        return (np.concatenate(sx).astype(np.float32), np.array(sy),
                np.concatenate(qx).astype(np.float32), np.array(qy))

    if task == "tabular":
        pools = _tabular_pools(seed=0)
        ds = ["wine", "iris"][rng.randint(2)]
        keys = [(ds, c) for c in range(3)]
        # exemplar split 80/20 with offset per split
        sx, sy, qx, qy = [], [], [], []
        for li, k in enumerate(keys[:n_way]):
            arr = pools[k]
            n = len(arr)
            cut = int(0.8 * n)
            part = arr[:cut] if split == "train" else arr[cut:]
            need = k_shot + q_query
            idx = rng.choice(len(part), need, replace=len(part) < need)
            samp = part[idx]
            sx.append(samp[:k_shot]); sy += [li] * k_shot
            qx.append(samp[k_shot:]); qy += [li] * q_query
        return (np.concatenate(sx).astype(np.float32), np.array(sy),
                np.concatenate(qx).astype(np.float32), np.array(qy))

    if task == "chess":
        X, y = _chess_dataset(seed=0)
        # exemplar split 80/20 stratified
        out = []
        for split_want in [split]:
            pass
        sx, sy, qx, qy = [], [], [], []
        for cls in range(2):
            arr = X[y == cls]
            n = len(arr); cut = int(0.8 * n)
            part = arr[:cut] if split == "train" else arr[cut:]
            idx = rng.choice(len(part), (k_shot + q_query), replace=False)
            samp = part[idx]
            sx.append(samp[:k_shot]); sy += [cls] * k_shot
            qx.append(samp[k_shot:]); qy += [cls] * q_query
        # binary labels already 0/1 == local ids
        return (np.concatenate(sx).astype(np.float32), np.array(sy),
                np.concatenate(qx).astype(np.float32), np.array(qy))

    if task == "coding":
        pools = _coding_pairs(seed=0)
        train_ids = [0, 1, 3, 4, 5, 6]
        test_ids = [2, 7, 8, 9]
        pool = train_ids if split == "train" else test_ids
        chosen = [pool[i] for i in rng.choice(len(pool), n_way, replace=False)]
        sx, sy, qx, qy = [], [], [], []
        for li, tid in enumerate(chosen):
            arr = pools[tid]
            idx = rng.choice(len(arr), k_shot + q_query, replace=False)
            samp = arr[idx]
            sx.append(samp[:k_shot]); sy += [li] * k_shot
            qx.append(samp[k_shot:]); qy += [li] * q_query
        return (np.concatenate(sx).astype(np.float32), np.array(sy),
                np.concatenate(qx).astype(np.float32), np.array(qy))

    raise ValueError(task)
