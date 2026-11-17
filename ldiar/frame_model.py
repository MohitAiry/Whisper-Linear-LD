"""Step 1c model: frame-level language tagger on frozen encoder features, and its Viterbi decoder.

Input (B, T, D) features at 25 Hz -> (B, T, NCLS) logits, class 0 = non-speech. No VAD and no boundary
detector: language changes of any density (DISPLACE: median same-language stretch 1.1 s) come out of the
frame labels, and the decoder only adds a switch penalty.
"""
import numpy as np
import torch
import torch.nn as nn


class FrameTagger(nn.Module):
    """layers >= 1: BiLSTM tagger (the reported system). Ablations (2026-11-13), same interface:
    layers = 0: per-frame MLP (LayerNorm, Dropout, Linear d_in->d, GELU, Dropout, Linear d->ncls), no context;
    layers = -1: linear probe (LayerNorm, Dropout, Linear d_in->ncls)."""

    def __init__(self, d_in, ncls, d=256, layers=2, dropout=0.2):
        super().__init__()
        self.layers = layers
        if layers < 0:
            self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, ncls))
            return
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, d), nn.GELU())
        self.rnn = nn.LSTM(d, d, num_layers=layers, batch_first=True, bidirectional=True,
                           dropout=dropout) if layers > 0 else None
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d if layers > 0 else d, ncls))

    def forward(self, x):
        if self.layers < 0:
            return self.inp(x)
        h = self.inp(x)
        if self.rnn is not None:
            h, _ = self.rnn(h)
        return self.out(h)

    @torch.no_grad()
    def session_logprobs(self, x, chunk=3000, ctx=250):
        """Whole session (T, D) -> (T, C) log-probabilities, in chunks of `chunk` frames with `ctx` frames of
        context on each side (so the BiLSTM sees inputs of about its training length)."""
        T, out = x.shape[0], []
        for s in range(0, T, chunk):
            lo, hi = max(0, s - ctx), min(T, s + chunk + ctx)
            lp = torch.log_softmax(self(x[None, lo:hi]).float()[0], -1)
            out.append(lp[s - lo:s - lo + min(chunk, T - s)])
        return torch.cat(out)


def viterbi(lp, penalty, allowed=None):
    """lp (T, C) log-probabilities (numpy). Every change of state costs `penalty` (nats).
    allowed: bool (C,) mask of states that may be used (class 0, non-speech, should stay allowed)."""
    T, C = lp.shape
    lp = lp.copy()
    if allowed is not None:
        lp[:, ~allowed] = -1e9
    back = np.zeros((T, C), np.int32)
    score = lp[0].copy()
    for t in range(1, T):
        best = int(score.argmax())
        stay, jump = score, score[best] - penalty
        use_jump = jump > stay
        back[t] = np.where(use_jump, best, np.arange(C))
        score = np.where(use_jump, jump, stay) + lp[t]
    path = np.empty(T, np.int32)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_multi(lp, penalties, allowed=None):
    """viterbi() for several switch penalties in one pass -> int32 (len(penalties), T), identical paths
    (same float32 arithmetic and tie-breaking: first argmax, a jump only if strictly better)."""
    T, C = lp.shape
    lp = lp.astype(np.float32, copy=True)
    if allowed is not None:
        lp[:, ~allowed] = -1e9
    pen = np.asarray(penalties, np.float32)[:, None]
    P, rows, states = len(pen), np.arange(len(pen)), np.arange(C)
    back = np.zeros((T, P, C), np.int8)
    score = np.repeat(lp[:1], P, 0)
    for t in range(1, T):
        best = score.argmax(1)
        jump = score[rows, best][:, None] - pen
        use_jump = jump > score
        back[t] = np.where(use_jump, best[:, None], states)
        score = np.where(use_jump, jump, score) + lp[t]
    path = np.empty((P, T), np.int32)
    path[:, -1] = score.argmax(1)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = back[t, rows, path[:, t]]
    return path


def session_languages(lp, min_share):
    """Languages kept for a session: those taking >= min_share of the speech frames by argmax."""
    C = lp.shape[1]
    am = lp.argmax(1)
    sp = am[am > 0]
    allowed = np.ones(C, bool)
    if min_share > 0 and len(sp):
        share = np.bincount(sp, minlength=C) / len(sp)
        allowed[1:] = share[1:] >= min_share
        if not allowed[1:].any():
            allowed[1 + int(share[1:].argmax())] = True
    return allowed


def path_to_segments(path, names, hop=0.04):
    """-> [(start s, end s, label)] for runs of non-zero classes."""
    segs, s = [], 0
    for t in range(1, len(path) + 1):
        if t == len(path) or path[t] != path[s]:
            if path[s] > 0:
                segs.append((s * hop, t * hop, names[path[s]]))
            s = t
    return segs
