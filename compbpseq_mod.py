from __future__ import annotations

import math
import re
from typing import Optional

import torch


def read_bpseq(file: str) -> tuple[str, list[int], Optional[str], Optional[float], Optional[float]]:
    with open(file) as f:
        p = [0]
        s = ['']
        name = sc = t = None
        for l in f:
            if l.startswith('#'):
                m = re.search(r'^# (.*) \(s=([\d.]+), ([\d.]+)s\)', l)
                if m:
                    name, sc, t = m[1], float(m[2]), float(m[3])

            else:
                idx, c, pair = l.rstrip('\n').split()
                s.append(c)
                p.append(int(pair))
    seq = ''.join(s)
    return (seq, p, name, sc, t)

def read_pdb(file: str) -> list[tuple[int, int]]:
    p = []
    with open(file) as f:
        for l in f:
            l = l.rstrip('\n').split()
            if len(l) == 2 and l[0].isdecimal() and l[1].isdecimal():
                p.append((int(l[0]), int(l[1])))
    return p

def compare_bpseq(ref, pred, seq) -> tuple[int, int, int, int]: #seqを関数内で使えるように追加
    #print(seq)
    #L = len(ref) - 1
    tp = fp = fn = tn = 0
    if ((len(ref)>0 and isinstance(ref[0], list)) or (isinstance(ref, torch.Tensor) and ref.ndim==2)):
        if isinstance(ref, torch.Tensor):
            ref = ref.tolist()
        ref = {(min(i, j), max(i, j)) for i, j in ref}
        pred = {(i, j) for i, j in enumerate(pred) if i < j}
        tp = len(ref & pred)
        fp = len(pred - ref)
        fn = len(ref - pred)
    else:
        assert(len(ref) == len(pred))
        for i, (j1, j2) in enumerate(zip(ref, pred)):
            #print(seq[i-1], j1, j2)
            if seq[i-1] == "I": #Iの場合
                #print(i, seq[i-1], j1, j2)
                if j1 > 0: # pos, i < j1の条件を消した(tnのため)
                    if j1 == j2:
                        tp += 1
                    elif j2 > 0 and i < j2:
                        fp += 1
                        fn += 1
                    else:
                        fn += 1
                elif j2 > 0 and i < j2:
                    fp += 1
                elif i > 0 and j1 == 0 and j2 == 0: #tnの条件
                    tn += 1
            elif seq[i-1] == "P":
                print(i, seq[i-1], j1, j2)
                if j1 > 0: # pos
                    if j1 == j2:
                        tp += 1
                    elif j2 > 0 and i < j2:
                        fp += 1
                        fn += 1
                    else:
                        fn += 1
                elif j2 > 0 and i < j2:
                    fp += 1
                elif j1 == 0 and j2 == 0:
                    tn += 1
            elif seq[i-1] == "M":
                print(i, seq[i-1], j1, j2)
                if j1 > 0: # pos
                    if j1 == j2:
                        tp += 1
                    elif j2 > 0 and i < j2:
                        fp += 1
                        fn += 1
                    else:
                        fn += 1
                elif j2 > 0 and i < j2:
                    fp += 1
                elif j1 == 0 and j2 == 0:
                    tn += 1
    #tn = L * (L - 1) // 2 - tp - fp - fn　←ここは必要なくてよい？
    return (tp, tn, fp, fn)

def accuracy(tp: int, tn: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    sen = tp / (tp + fn) if tp+fn > 0. else 0.
    ppv = tp / (tp + fp) if tp+fp > 0. else 0.
    fval = 2 * sen * ppv / (sen + ppv) if sen+ppv > 0. else 0.
    mcc = ((tp*tn)-(fp*fn)) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0. else 0.
    return (sen, ppv, fval, mcc)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='calculate SEN, PPV, F, MCC for the predicted RNA secondary structure', add_help=True)
    parser.add_argument('ref', type=str, help='BPSEQ-formatted file with the refernece structure')
    parser.add_argument('pred', type=str, help='BPSEQ-formatted file with the predicted structure')
    parser.add_argument('--pdb', action='store_true', help='use pdb labels for ref')
    args = parser.parse_args()
    if args.pdb:
        ref = read_pdb(args.ref)
    else:
        seq, ref, _, _, _ = read_bpseq(args.ref)
    seq, pred, name, sc, t = read_bpseq(args.pred)
    x = compare_bpseq(ref, pred, seq) #compare_bpseq内でseqの情報を使いたかったので、seqを追加した。
    x = [name, len(seq), t, sc] + list(x) + list(accuracy(*x))
    print(', '.join([str(v) for v in x]))
