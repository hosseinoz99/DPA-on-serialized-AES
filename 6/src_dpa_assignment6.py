
# src_dpa_assignment6.py
# Fully commented, vectorized DPA on serialized AES for Assignment 6.
# Student: Omidizadeh
#
# Usage:
#   python src_dpa_assignment6.py
#
# Requirements:
#   - NumPy, Matplotlib
#   - Files in current working dir or /mnt/data:
#       Traces00000.dat, plaintexts.dat
#
import os, numpy as np, matplotlib.pyplot as plt

BASE = os.path.dirname(__file__) if os.path.dirname(__file__) else "."
def _p(name):
    p_local = os.path.join(BASE, name)
    return p_local if os.path.exists(p_local) else os.path.join("/mnt/data", name)

TRACES_PATH = _p("Traces00000.dat")
PTS_PATH    = _p("plaintexts.dat")

N_TRACES  = 10_000
N_SAMPLES = 10_000
STATE_LEN = 16

# AES S-box
SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
], dtype=np.uint8)

def load_traces():
    tr_raw = np.fromfile(TRACES_PATH, dtype=np.int8)
    assert tr_raw.size == N_TRACES * N_SAMPLES, "Unexpected traces file size"
    return tr_raw.reshape(N_TRACES, N_SAMPLES).astype(np.float32)

def load_pts():
    pts_raw = np.fromfile(PTS_PATH, dtype=np.uint8)
    assert pts_raw.size == N_TRACES * STATE_LEN, "Unexpected plaintexts size"
    return pts_raw.reshape(STATE_LEN, N_TRACES)

def pick_pois(traces_mat, k=16, guard=25):
    var = traces_mat.var(axis=0)
    idxs, vc = [], var.copy()
    for _ in range(k):
        j = int(np.argmax(vc)); idxs.append(j)
        lo = max(0, j-guard); hi = min(vc.size, j+guard+1)
        vc[lo:hi] = -np.inf
    idxs.sort(); return np.array(idxs, dtype=int)

def bit_sign_table(sb_mat, bit):
    bj = ((sb_mat >> bit) & 1).astype(np.int8)
    s = (1 - 2*bj).astype(np.float32)
    return s.T

def recover_keys(traces, p5, p10, n_sub=2000, n_poi=16):
    all_k = np.arange(256, dtype=np.uint16)
    sub_idx = np.arange(n_sub)
    tr_sub = traces[sub_idx]
    p5_sub = p5[sub_idx]
    p10_sub= p10[sub_idx]
    T_POI = pick_pois(tr_sub, k=n_poi, guard=25)
    tr_sub_poi = tr_sub[:, T_POI]  # (n_sub, n_poi)
    V = tr_sub_poi.T               # (n_poi, n_sub)

    SBOX = globals()["SBOX"]
    sb_p5 = SBOX[p5_sub[:, None] ^ all_k[None, :]]
    sb_p10= SBOX[p10_sub[:, None] ^ all_k[None, :]]

    scores = np.zeros((256,256), dtype=np.float32)
    win_bit, win_poi = 0, 0
    for bit in range(8):
        X = bit_sign_table(sb_p5, bit)
        Y = bit_sign_table(sb_p10, bit)
        s_sum = X @ Y.T
        n1 = (n_sub - s_sum) * 0.5
        n0 = (n_sub + s_sum) * 0.5
        denom = np.sqrt(np.maximum(n0, 1.0) * np.maximum(n1, 1.0))
        for pi in range(V.shape[0]):
            v = V[pi]
            B = v[:, None] * Y.T
            M = X @ B
            sc = np.abs(M) / denom
            imp = sc > scores
            if np.any(imp):
                scores[imp] = sc[imp]
                win_bit, win_poi = bit, pi
    best_idx = int(np.argmax(scores))
    best_k5 = best_idx // 256
    best_k10= best_idx % 256
    return best_k5, best_k10, scores, win_bit, int(win_poi), T_POI

def tcurve_for_pair_full(traces, p5, p10, k5i, k10i):
    # choose best bit by coarse scan
    S = traces[:, ::25]
    SBOX = globals()["SBOX"]
    sb5 = SBOX[p5 ^ np.uint8(k5i)]
    sb10= SBOX[p10 ^ np.uint8(k10i)]
    best_bit, best_peak = 0, -1.0
    for b in range(8):
        gg = (((sb5 >> b) & 1) ^ ((sb10 >> b) & 1)).astype(np.uint8)
        A = S[gg == 1]; B = S[gg == 0]
        nA = max(1, A.shape[0]); nB = max(1, B.shape[0])
        mA = A.mean(axis=0); mB = B.mean(axis=0)
        vA = A.var(axis=0, ddof=1) if nA > 1 else np.zeros_like(mA)
        vB = B.var(axis=0, ddof=1) if nB > 1 else np.zeros_like(mB)
        tcv = (mA - mB) / np.sqrt(np.maximum(vA / nA + vB / nB, 1e-12))
        peak = np.max(np.abs(tcv))
        if peak > best_peak:
            best_peak = float(peak); best_bit = b
    gg = (((sb5 >> best_bit) & 1) ^ ((sb10 >> best_bit) & 1)).astype(np.uint8)
    A = traces[gg == 1]; B = traces[gg == 0]
    nA = max(1, A.shape[0]); nB = max(1, B.shape[0])
    mA = A.mean(axis=0); mB = B.mean(axis=0)
    vA = A.var(axis=0, ddof=1) if nA > 1 else np.zeros_like(mA)
    vB = B.var(axis=0, ddof=1) if nB > 1 else np.zeros_like(mB)
    tcv = (mA - mB) / np.sqrt(np.maximum(vA / nA + vB / nB, 1e-12))
    return tcv

def main():
    traces = load_traces()
    pts = load_pts()
    p5  = pts[5]; p10 = pts[10]

    k5, k10, scores, win_bit, win_poi, T_POI = recover_keys(traces, p5, p10, n_sub=2000, n_poi=16)
    print(f"Recovered: k5={k5} (0x{k5:02x}), k10={k10} (0x{k10:02x})")

    # plot_over_kg.png
    fig1 = plt.figure(figsize=(10,4), dpi=150)
    plt.plot(scores.reshape(-1))
    plt.title("Score over all 2^16 key guesses (max |t|-like over bits & POIs)")
    plt.xlabel("Key index (k5*256 + k10)")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("plot_over_kg.png", bbox_inches="tight")
    plt.close(fig1)

    # plot_over_traces.png: recovered vs a few impostors
    scores_lin = scores.reshape(-1)
    order = np.argsort(scores_lin)[::-1]

    t_best = np.abs(tcurve_for_pair_full(traces, p5, p10, k5, k10))

    fig2 = plt.figure(figsize=(10,4), dpi=150)
    plt.plot(t_best, label=f"best ({k5},{k10})")
    impostors = 5
    count = 0
    for idx in order:
        k5i = idx // 256; k10i = idx % 256
        if k5i == k5 and k10i == k10: continue
        t_imp = np.abs(tcurve_for_pair_full(traces, p5, p10, k5i, k10i))
        plt.plot(t_imp, alpha=0.7, label=f"impostor {count+1}")
        count += 1
        if count >= impostors: break
    plt.title("|t|-curve over time samples")
    plt.xlabel("Sample index")
    plt.ylabel("|t|")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_over_traces.png", bbox_inches="tight")
    plt.close(fig2)

    # keys.txt
    with open("keys.txt","w") as f:
        f.write(f"k5  : dec={k5} hex=0x{k5:02x}\n")
        f.write(f"k10 : dec={k10} hex=0x{k10:02x}\n")

if __name__ == "__main__":
    main()
