"""
A-DAEC V3
Adaptive Duplicate-Aware Event Commitment for Efficient IoT Intrusion Auditing

Primary research objective
--------------------------
Minimize blockchain commitments subject to:
    1) Attack homogeneity = 100%
    2) Semantic signature homogeneity >= TARGET_SEMANTIC_HOMOGENEITY
    3) Audit coverage = 100%
    4) Verification success = 100%
    5) Tamper detection = 100%
    6) Maximum group size <= MAX_GROUP_SIZE

The proposed A-DAEC uses ONLY semantic/network fields for grouping.
The Independent Behavioral Cohesion Score (IBCS) uses different numerical
features and is therefore NOT used to create A-DAEC groups.

Baselines
---------
1) Full Anchoring
2) Fixed-Batch
3) Class-Based
4) DAEC V2
5) Proposed A-DAEC V3

Place beside:
    train_test_network.csv

Output folder:
    adaec_results_v3
"""

import sys
import time
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy are required.")
    print("Run in Anaconda Prompt:")
    print("conda activate calt_ids")
    print("conda install pandas numpy -y")
    input("Press Enter to finish...")
    sys.exit(1)


# =====================================================================
# CONFIGURATION
# =====================================================================

DATASET_NAME = "train_test_network.csv"
RESULTS_DIR = "adaec_results_v3"

SAMPLE_SIZES = [10000, 25000, 50000, 100000]
INCLUDE_FULL_DATASET = True
RANDOM_SEED = 42

MAX_GROUP_SIZE = 100

# Main A-DAEC semantic constraint.
TARGET_SEMANTIC_HOMOGENEITY = 0.90

# Full-dataset sensitivity analysis.
SENSITIVITY_THRESHOLDS = [0.80, 0.85, 0.90, 0.95]

# Runtime/reproducibility.
TIMING_REPEATS = 5
VERIFICATION_SAMPLES = 1000

# Logical on-chain record estimate:
# commitment + previous hash + block hash + compact metadata.
ONCHAIN_RECORD_BYTES = 120

ATTACK_COLUMN_CANDIDATES = ["type", "attack", "attack_type", "category"]
BINARY_LABEL_CANDIDATES = ["label"]

# Core semantic signature used for the explicit semantic-homogeneity constraint.
# These fields create the "duplicate-aware" network/security signature.
CORE_SIGNATURE_COLUMNS = [
    "proto",
    "service",
    "dst_port",
    "conn_state",
]

# Rich semantic ordering fields.
# IMPORTANT: numerical behavioral fields are intentionally excluded.
# A-DAEC uses these fields only to place semantically related events adjacent
# before adaptive constrained packing.
RICH_SEMANTIC_ORDER_COLUMNS = [
    "proto",
    "service",
    "dst_port",
    "conn_state",
    "dns_query",
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "dns_AA",
    "dns_RD",
    "dns_RA",
    "dns_rejected",
    "ssl_version",
    "ssl_cipher",
    "ssl_resumed",
    "ssl_established",
    "http_trans_depth",
    "http_method",
    "http_version",
    "http_status_code",
    "weird_name",
    "weird_notice",
]

# Independent numerical features: NEVER used by A-DAEC grouping.
# They are used only to evaluate whether groups are behaviorally coherent
# beyond the semantic fields used by the algorithm.
BEHAVIORAL_EVALUATION_COLUMNS = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "dst_pkts",
    "src_ip_bytes",
    "dst_ip_bytes",
]


# =====================================================================
# CRYPTOGRAPHIC FUNCTIONS
# =====================================================================

def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def merkle_root(leaves):
    if not leaves:
        return sha256_bytes(b"")
    if len(leaves) == 1:
        return leaves[0]

    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            sha256_bytes(level[i] + level[i + 1])
            for i in range(0, len(level), 2)
        ]
    return level[0]


def merkle_proof(leaves, target_index):
    if not leaves:
        return [], sha256_bytes(b"")

    level = list(leaves)
    idx = int(target_index)
    proof = []

    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])

        if idx % 2 == 0:
            sibling_idx = idx + 1
            sibling_is_left = False
        else:
            sibling_idx = idx - 1
            sibling_is_left = True

        proof.append((level[sibling_idx], sibling_is_left))

        level = [
            sha256_bytes(level[i] + level[i + 1])
            for i in range(0, len(level), 2)
        ]
        idx //= 2

    return proof, level[0]


def verify_merkle_proof(leaf_hash, proof, expected_root):
    current = leaf_hash
    for sibling_hash, sibling_is_left in proof:
        if sibling_is_left:
            current = sha256_bytes(sibling_hash + current)
        else:
            current = sha256_bytes(current + sibling_hash)
    return current == expected_root


# =====================================================================
# DATA PREPARATION
# =====================================================================

def find_first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def deterministic_stratified_sample(df, n, class_col, seed):
    """Exact-size proportional stratified sample + deterministic shuffle."""
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if class_col is None:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    counts = df[class_col].value_counts(dropna=False)
    raw = counts / len(df) * n
    quotas = np.floor(raw).astype(int)

    remaining = int(n - quotas.sum())
    if remaining > 0:
        fractional = (raw - quotas).sort_values(ascending=False)
        for cls in fractional.index[:remaining]:
            quotas.loc[cls] += 1

    parts = []
    for i, (cls, q) in enumerate(quotas.items()):
        if q <= 0:
            continue
        if pd.isna(cls):
            subset = df[df[class_col].isna()]
        else:
            subset = df[df[class_col] == cls]
        q = min(int(q), len(subset))
        parts.append(subset.sample(n=q, random_state=seed + i + 1))

    sample = pd.concat(parts, axis=0)

    if len(sample) < n:
        missing = n - len(sample)
        remaining_df = df.drop(index=sample.index)
        extra = remaining_df.sample(
            n=missing, random_state=seed + 777
        )
        sample = pd.concat([sample, extra], axis=0)

    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed + 778)

    return sample.sample(
        frac=1.0, random_state=seed + 1000
    ).reset_index(drop=True)


def build_event_hashes(df):
    """
    Hash the COMPLETE original event.
    All 44 input fields therefore contribute to event integrity.
    """
    hashes = []
    append_hash = hashes.append
    sep = "\x1f"

    for row in df.itertuples(index=False, name=None):
        payload = sep.join(
            "<NA>" if pd.isna(v) else str(v) for v in row
        )
        append_hash(
            hashlib.sha256(payload.encode("utf-8")).digest()
        )

    return hashes


def tuple_keys(df, columns):
    if not columns:
        return [("<NONE>",)] * len(df)

    safe = df[columns].fillna("<NA>").astype(str)
    return list(safe.itertuples(index=False, name=None))


def prepare_behavior_matrix(df, columns):
    """
    Independent behavioral evaluation matrix:
    log1p + robust median/IQR scaling.

    These columns are not used by A-DAEC grouping.
    """
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return np.zeros((len(df), 1), dtype=float), []

    x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    arr = x.to_numpy(dtype=float)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    arr = np.log1p(arr)

    median = np.nanmedian(arr, axis=0)
    q1 = np.nanpercentile(arr, 25, axis=0)
    q3 = np.nanpercentile(arr, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr < 1e-9] = 1.0

    z = (arr - median) / iqr
    return z, cols


# =====================================================================
# GROUPING METHODS
# =====================================================================

def chunk_indices(indices, size):
    return [
        indices[i:i + size]
        for i in range(0, len(indices), size)
    ]


def groups_full(df, attack_col, core_cols, rich_cols, threshold):
    return [[i] for i in range(len(df))]


def groups_fixed(df, attack_col, core_cols, rich_cols, threshold):
    return [
        list(range(i, min(i + MAX_GROUP_SIZE, len(df))))
        for i in range(0, len(df), MAX_GROUP_SIZE)
    ]


def groups_class_based(df, attack_col, core_cols, rich_cols, threshold):
    if attack_col is None:
        return groups_fixed(
            df, attack_col, core_cols, rich_cols, threshold
        )

    groups = []
    labels = df[attack_col].astype(str).tolist()
    buckets = defaultdict(list)

    for idx, label in enumerate(labels):
        buckets[label].append(idx)

    for indices in buckets.values():
        groups.extend(chunk_indices(indices, MAX_GROUP_SIZE))

    return groups


def groups_daec_v2(df, attack_col, core_cols, rich_cols, threshold):
    """
    DAEC V2 baseline:
    exact (attack + core semantic signature) grouping, capped at 100.
    """
    if attack_col is None or not core_cols:
        return groups_class_based(
            df, attack_col, core_cols, rich_cols, threshold
        )

    work_cols = [attack_col] + core_cols
    safe = df[work_cols].fillna("<NA>").astype(str)

    buckets = defaultdict(list)
    for idx, key in enumerate(
        safe.itertuples(index=False, name=None)
    ):
        buckets[key].append(idx)

    groups = []
    for indices in buckets.values():
        groups.extend(chunk_indices(indices, MAX_GROUP_SIZE))

    return groups


def prepare_adaec_order(df, attack_col, rich_cols):
    """
    Compute A-DAEC semantic ordering ONCE per sample.
    This is deterministic method-specific preprocessing and its time is
    reported separately so repeated timing does not redo the same sort.
    """
    available_rich = [c for c in rich_cols if c in df.columns]
    if attack_col is None:
        raise ValueError("A-DAEC requires an attack/category column.")

    cols = [attack_col] + available_rich
    semantic = df[cols].fillna("<NA>").astype(str)
    ordered_by_attack = []

    for _, attack_indices in semantic.groupby(attack_col, sort=False).groups.items():
        sub = semantic.loc[attack_indices]
        if available_rich:
            ordered = sub.sort_values(
                available_rich, kind="mergesort"
            ).index.to_numpy(dtype=np.int32)
        else:
            ordered = np.asarray(list(attack_indices), dtype=np.int32)
        ordered_by_attack.append(ordered)

    return ordered_by_attack


def groups_adaec_prepared(ordered_by_attack, core_signatures, threshold):
    """
    Adaptive Semantic-Constrained Packing using a precomputed semantic order.

    The algorithm keeps each attack class separate and fills groups up to
    MAX_GROUP_SIZE only while the predicted core-signature purity remains
    at or above the requested threshold.
    """
    groups = []

    for ordered_indices in ordered_by_attack:
        current_group = []
        sig_counts = Counter()
        dominant_count = 0

        for idx_np in ordered_indices:
            idx = int(idx_np)
            signature = core_signatures[idx]

            if not current_group:
                current_group = [idx]
                sig_counts = Counter({signature: 1})
                dominant_count = 1
                continue

            if len(current_group) >= MAX_GROUP_SIZE:
                groups.append(current_group)
                current_group = [idx]
                sig_counts = Counter({signature: 1})
                dominant_count = 1
                continue

            candidate_count = sig_counts.get(signature, 0) + 1
            candidate_dominant = max(dominant_count, candidate_count)
            candidate_size = len(current_group) + 1
            candidate_purity = candidate_dominant / candidate_size

            if candidate_purity >= threshold:
                current_group.append(idx)
                sig_counts[signature] = candidate_count
                dominant_count = candidate_dominant
            else:
                groups.append(current_group)
                current_group = [idx]
                sig_counts = Counter({signature: 1})
                dominant_count = 1

        if current_group:
            groups.append(current_group)

    return groups


METHOD_RUNNERS = {
    "Full Anchoring": groups_full,
    "Fixed-Batch": groups_fixed,
    "Class-Based": groups_class_based,
    "DAEC V2": groups_daec_v2,
}


# =====================================================================
# BLOCKCHAIN + MERKLE COMMITMENT
# =====================================================================

def build_commitments(groups, event_hashes):
    """
    Build one Merkle commitment per group and chain commitments
    through SHA-256 previous-block linking. No mining / PoW.
    """
    roots = []
    event_group = np.empty(
        len(event_hashes), dtype=np.int32
    )
    event_position = np.empty(
        len(event_hashes), dtype=np.int16
    )

    previous_hash = bytes(32)

    for gid, group in enumerate(groups):
        leaves = [event_hashes[i] for i in group]
        root = merkle_root(leaves)
        roots.append(root)

        header = (
            previous_hash
            + root
            + gid.to_bytes(8, "big", signed=False)
            + len(group).to_bytes(8, "big", signed=False)
        )
        previous_hash = sha256_bytes(header)

        for pos, event_idx in enumerate(group):
            event_group[event_idx] = gid
            event_position[event_idx] = pos

    return roots, event_group, event_position


# =====================================================================
# METRICS
# =====================================================================

def weighted_homogeneity(groups, values):
    if not groups:
        return 0.0

    matched = 0
    total = 0

    for group in groups:
        counts = Counter(values[i] for i in group)
        matched += counts.most_common(1)[0][1]
        total += len(group)

    return 100.0 * matched / total if total else 0.0


def behavioral_cohesion_score(groups, behavior_z):
    """
    Independent Behavioral Cohesion Score (IBCS), higher is better.

    For each group:
      - robust group center = median vector
      - event dispersion = mean absolute robust-z deviation
    Global weighted mean dispersion is converted to:
        IBCS = 100 / (1 + mean_dispersion)

    Because behavioral features are not used by A-DAEC grouping,
    this is an independent external quality test.
    """
    if len(behavior_z) == 0:
        return 0.0, 0.0

    # Singleton groups have zero within-group dispersion.
    if all(len(g) == 1 for g in groups):
        return 100.0, 0.0

    total_distance = 0.0
    total_events = 0

    for group in groups:
        arr = behavior_z[group]
        center = np.median(arr, axis=0)
        event_dist = np.abs(arr - center).mean(axis=1)
        total_distance += float(event_dist.sum())
        total_events += len(group)

    mean_dispersion = (
        total_distance / total_events
        if total_events else 0.0
    )
    score = 100.0 / (1.0 + mean_dispersion)
    return score, mean_dispersion


def evaluate_verification(
    groups,
    roots,
    event_group,
    event_position,
    event_hashes,
    seed
):
    n = len(event_hashes)
    if n == 0:
        return {
            "Verification_Time_ms": 0.0,
            "Verification_Success_pct": 0.0,
            "Tamper_Detection_pct": 0.0,
            "Avg_Merkle_Proof_bytes": 0.0,
        }

    k = min(VERIFICATION_SAMPLES, n)
    rng = np.random.default_rng(seed)
    selected = rng.choice(n, size=k, replace=False)

    success = 0
    tamper_detected = 0
    proof_bytes = 0

    start = time.perf_counter()

    for event_idx in selected:
        gid = int(event_group[event_idx])
        pos = int(event_position[event_idx])
        group = groups[gid]
        leaves = [event_hashes[i] for i in group]

        proof, expected_root = merkle_proof(leaves, pos)

        if verify_merkle_proof(
            event_hashes[event_idx],
            proof,
            expected_root
        ):
            success += 1

        tampered_hash = sha256_bytes(
            b"TAMPERED|" + event_hashes[event_idx]
        )

        if not verify_merkle_proof(
            tampered_hash,
            proof,
            expected_root
        ):
            tamper_detected += 1

        proof_bytes += len(proof) * 32

    elapsed = time.perf_counter() - start

    return {
        "Verification_Time_ms": elapsed / k * 1000.0,
        "Verification_Success_pct": success / k * 100.0,
        "Tamper_Detection_pct": (
            tamper_detected / k * 100.0
        ),
        "Avg_Merkle_Proof_bytes": proof_bytes / k,
    }


def run_method(
    method_name,
    df,
    event_hashes,
    attack_col,
    core_cols,
    rich_cols,
    attack_values,
    core_signatures,
    behavior_z,
    threshold,
    seed,
    adaec_order=None,
    method_preparation_time=0.0,
):
    runner = METHOD_RUNNERS.get(method_name)

    timing_values = []
    final_groups = None
    final_roots = None
    final_event_group = None
    final_event_position = None

    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()

        if method_name == "A-DAEC V3":
            if adaec_order is None:
                raise ValueError("A-DAEC semantic order was not prepared.")
            groups = groups_adaec_prepared(
                adaec_order, core_signatures, threshold
            )
        else:
            groups = runner(
                df,
                attack_col,
                core_cols,
                rich_cols,
                threshold
            )

        roots, event_group, event_position = (
            build_commitments(groups, event_hashes)
        )

        timing_values.append(
            time.perf_counter() - start
        )

        final_groups = groups
        final_roots = roots
        final_event_group = event_group
        final_event_position = event_position

    processing_mean = float(
        np.mean(timing_values)
    )
    processing_sd = float(
        np.std(timing_values, ddof=1)
        if len(timing_values) > 1 else 0.0
    )

    tx = len(final_groups)
    covered = sum(len(g) for g in final_groups)

    attack_hom = weighted_homogeneity(
        final_groups, attack_values
    )
    semantic_hom = weighted_homogeneity(
        final_groups, core_signatures
    )

    ibcs, behavior_dispersion = (
        behavioral_cohesion_score(
            final_groups, behavior_z
        )
    )

    verify = evaluate_verification(
        final_groups,
        final_roots,
        final_event_group,
        final_event_position,
        event_hashes,
        seed
    )

    group_sizes = [len(g) for g in final_groups]

    result = {
        "Method": method_name,
        "Events": len(df),
        "Transactions": tx,
        "Estimated_OnChain_Storage_bytes": (
            tx * ONCHAIN_RECORD_BYTES
        ),
        "Method_Preparation_Time_s": float(method_preparation_time),
        "Processing_Time_Mean_s": processing_mean,
        "Processing_Time_SD_s": processing_sd,
        "Algorithm_Time_Mean_s": float(method_preparation_time) + processing_mean,
        "Throughput_events_s": (
            len(df) / processing_mean
            if processing_mean > 0 else float("inf")
        ),
        "Audit_Coverage_pct": (
            covered / len(df) * 100.0
            if len(df) else 0.0
        ),
        "Attack_Homogeneity_pct": attack_hom,
        "Semantic_Homogeneity_pct": semantic_hom,
        "IBCS_pct": ibcs,
        "Behavioral_Dispersion": behavior_dispersion,
        "Avg_Group_Size": (
            float(np.mean(group_sizes))
            if group_sizes else 0.0
        ),
        "Max_Group_Size": (
            max(group_sizes) if group_sizes else 0
        ),
    }
    result.update(verify)

    # Explicit research constraint.
    eps = 1e-9
    result["Constraint_Feasible"] = (
        attack_hom >= 100.0 - eps
        and semantic_hom >= (
            TARGET_SEMANTIC_HOMOGENEITY * 100.0
        ) - eps
        and result["Audit_Coverage_pct"] >= 100.0 - eps
        and result["Verification_Success_pct"] >= 100.0 - eps
        and result["Tamper_Detection_pct"] >= 100.0 - eps
        and result["Max_Group_Size"] <= MAX_GROUP_SIZE
    )

    return result


# =====================================================================
# SENSITIVITY ANALYSIS
# =====================================================================

def run_adaec_sensitivity(
    df,
    event_hashes,
    attack_col,
    core_cols,
    rich_cols,
    attack_values,
    core_signatures,
    behavior_z,
    adaec_order,
):
    rows = []

    for threshold in SENSITIVITY_THRESHOLDS:
        start = time.perf_counter()

        groups = groups_adaec_prepared(
            adaec_order, core_signatures, threshold
        )

        elapsed = time.perf_counter() - start

        attack_hom = weighted_homogeneity(
            groups, attack_values
        )
        semantic_hom = weighted_homogeneity(
            groups, core_signatures
        )
        ibcs, dispersion = behavioral_cohesion_score(
            groups, behavior_z
        )

        tx = len(groups)

        rows.append({
            "Semantic_Constraint_pct": threshold * 100.0,
            "Transactions": tx,
            "TRR_pct": (
                1.0 - tx / len(df)
            ) * 100.0,
            "Attack_Homogeneity_pct": attack_hom,
            "Semantic_Homogeneity_pct": semantic_hom,
            "IBCS_pct": ibcs,
            "Behavioral_Dispersion": dispersion,
            "Grouping_Time_s": elapsed,
            "Avg_Group_Size": (
                len(df) / tx if tx else 0.0
            ),
        })

    return pd.DataFrame(rows)


# =====================================================================
# MAIN
# =====================================================================

def main():
    base = Path(__file__).resolve().parent
    dataset_path = base / DATASET_NAME
    results_dir = base / RESULTS_DIR
    results_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print(" A-DAEC V3 — Adaptive Duplicate-Aware Event Commitment")
    print("=" * 80)
    print(f"Python executable : {sys.executable}")
    print(f"Dataset           : {dataset_path}")

    if not dataset_path.exists():
        print(f"\nERROR: {DATASET_NAME} was not found.")
        print("Place the CSV in the same folder as this code.")
        input("\nPress Enter to finish...")
        return 1

    load_start = time.perf_counter()
    df = pd.read_csv(dataset_path)
    load_time = time.perf_counter() - load_start

    attack_col = find_first_existing(
        df.columns, ATTACK_COLUMN_CANDIDATES
    )
    label_col = find_first_existing(
        df.columns, BINARY_LABEL_CANDIDATES
    )

    core_cols = [
        c for c in CORE_SIGNATURE_COLUMNS
        if c in df.columns
    ]
    rich_cols = [
        c for c in RICH_SEMANTIC_ORDER_COLUMNS
        if c in df.columns
    ]
    behavior_cols = [
        c for c in BEHAVIORAL_EVALUATION_COLUMNS
        if c in df.columns
    ]

    print(f"Rows              : {len(df):,}")
    print(f"Columns           : {len(df.columns)}")
    print(f"CSV load time     : {load_time:.3f} s")
    print(f"Attack column     : {attack_col}")
    print(f"Binary label      : {label_col}")
    print(f"Core signature    : {core_cols}")
    print(f"Behavioral test   : {behavior_cols}")
    print(
        "Semantic target   : "
        f"{TARGET_SEMANTIC_HOMOGENEITY * 100:.1f}%"
    )
    print(f"Max group size    : {MAX_GROUP_SIZE}")
    print(f"Timing repeats    : {TIMING_REPEATS}")
    print(
        f"Verification tests: {VERIFICATION_SAMPLES:,} / method"
    )

    if attack_col is None:
        raise ValueError(
            "No attack-type column was found."
        )

    # Full distribution.
    distribution = (
        df[attack_col]
        .value_counts(dropna=False)
        .rename_axis("Attack_Type")
        .reset_index(name="Count")
    )
    distribution["Percentage"] = (
        distribution["Count"] / len(df) * 100.0
    )
    distribution.to_csv(
        results_dir / "dataset_attack_distribution_v3.csv",
        index=False
    )

    run_sizes = [
        n for n in SAMPLE_SIZES
        if n < len(df)
    ]
    if INCLUDE_FULL_DATASET:
        run_sizes.append(len(df))
    run_sizes = list(dict.fromkeys(run_sizes))

    methods = [
        "Full Anchoring",
        "Fixed-Batch",
        "Class-Based",
        "DAEC V2",
        "A-DAEC V3",
    ]

    all_results = []
    sample_distributions = []
    full_context = None

    for n in run_sizes:
        label = (
            "FULL" if n == len(df)
            else f"{n:,}"
        )

        print("\n" + "-" * 80)
        print(f" Sample size: {label}")
        print("-" * 80)

        sample_start = time.perf_counter()
        sample = deterministic_stratified_sample(
            df,
            n,
            attack_col,
            RANDOM_SEED + n % 1000
        )
        sampling_time = (
            time.perf_counter() - sample_start
        )

        print(
            f"Stratified sampling : {sampling_time:.3f} s"
        )

        dist = (
            sample[attack_col]
            .value_counts(dropna=False)
            .rename_axis("Attack_Type")
            .reset_index(name="Count")
        )
        dist["Percentage"] = (
            dist["Count"] / len(sample) * 100.0
        )
        dist["Sample_Size"] = n
        sample_distributions.append(dist)

        hash_start = time.perf_counter()
        event_hashes = build_event_hashes(sample)
        hashing_time = time.perf_counter() - hash_start
        print(
            f"Full-event SHA-256  : {hashing_time:.3f} s"
        )

        metric_start = time.perf_counter()

        attack_values = (
            sample[attack_col]
            .fillna("<NA>")
            .astype(str)
            .tolist()
        )

        core_signatures = tuple_keys(
            sample, core_cols
        )

        behavior_z, used_behavior_cols = (
            prepare_behavior_matrix(
                sample, behavior_cols
            )
        )

        metric_prep_time = (
            time.perf_counter() - metric_start
        )

        # A-DAEC semantic ordering is deterministic method-specific
        # preprocessing. Compute it once, report its cost separately,
        # and reuse it for repeated timing and sensitivity analysis.
        adaec_prep_start = time.perf_counter()
        adaec_order = prepare_adaec_order(
            sample, attack_col, rich_cols
        )
        adaec_preparation_time = (
            time.perf_counter() - adaec_prep_start
        )
        print(
            f"A-DAEC semantic prep : {adaec_preparation_time:.3f} s"
        )

        sample_results = []

        for method in methods:
            print(
                f"Running {method:<15} ... ",
                end="",
                flush=True
            )

            result = run_method(
                method,
                sample,
                event_hashes,
                attack_col,
                core_cols,
                rich_cols,
                attack_values,
                core_signatures,
                behavior_z,
                TARGET_SEMANTIC_HOMOGENEITY,
                RANDOM_SEED + n,
                adaec_order=(
                    adaec_order if method == "A-DAEC V3" else None
                ),
                method_preparation_time=(
                    adaec_preparation_time
                    if method == "A-DAEC V3" else 0.0
                ),
            )

            result["Sampling_Time_s"] = (
                sampling_time
            )
            result["Event_Hashing_Time_s"] = (
                hashing_time
            )
            result["Metric_Preparation_Time_s"] = (
                metric_prep_time
            )
            result["Total_EndToEnd_Mean_s"] = (
                hashing_time
                + metric_prep_time
                + result["Algorithm_Time_Mean_s"]
            )

            sample_results.append(result)

            print(
                f"{result['Transactions']:,} tx | "
                f"semantic={result['Semantic_Homogeneity_pct']:.2f}% | "
                f"IBCS={result['IBCS_pct']:.2f}% | "
                f"time={result['Algorithm_Time_Mean_s']:.3f}s"
            )

        full = next(
            r for r in sample_results
            if r["Method"] == "Full Anchoring"
        )

        for r in sample_results:
            r["TRR_pct"] = (
                1.0
                - r["Transactions"]
                / full["Transactions"]
            ) * 100.0

            r["SRR_pct"] = (
                1.0
                - r["Estimated_OnChain_Storage_bytes"]
                / full[
                    "Estimated_OnChain_Storage_bytes"
                ]
            ) * 100.0

        all_results.extend(sample_results)

        if n == len(df):
            full_context = (
                sample,
                event_hashes,
                attack_values,
                core_signatures,
                behavior_z,
                adaec_order,
            )

    results_df = pd.DataFrame(all_results)

    preferred = [
        "Events",
        "Method",
        "Transactions",
        "TRR_pct",
        "Estimated_OnChain_Storage_bytes",
        "SRR_pct",
        "Constraint_Feasible",
        "Attack_Homogeneity_pct",
        "Semantic_Homogeneity_pct",
        "IBCS_pct",
        "Behavioral_Dispersion",
        "Audit_Coverage_pct",
        "Verification_Success_pct",
        "Tamper_Detection_pct",
        "Verification_Time_ms",
        "Avg_Merkle_Proof_bytes",
        "Avg_Group_Size",
        "Max_Group_Size",
        "Method_Preparation_Time_s",
        "Processing_Time_Mean_s",
        "Processing_Time_SD_s",
        "Algorithm_Time_Mean_s",
        "Throughput_events_s",
        "Total_EndToEnd_Mean_s",
        "Sampling_Time_s",
        "Event_Hashing_Time_s",
        "Metric_Preparation_Time_s",
    ]
    results_df = results_df[
        [c for c in preferred if c in results_df.columns]
    ]

    results_path = (
        results_dir / "adaec_v3_results.csv"
    )
    results_df.to_csv(
        results_path, index=False
    )

    pd.concat(
        sample_distributions,
        ignore_index=True
    ).to_csv(
        results_dir
        / "sample_attack_distributions_v3.csv",
        index=False
    )

    # Sensitivity analysis only on the full dataset.
    sensitivity_path = None
    if full_context is not None:
        (
            full_sample,
            full_hashes,
            full_attack_values,
            full_core_signatures,
            full_behavior_z,
            full_adaec_order,
        ) = full_context

        print("\n" + "-" * 80)
        print(" A-DAEC full-dataset sensitivity analysis")
        print("-" * 80)

        sensitivity = run_adaec_sensitivity(
            full_sample,
            full_hashes,
            attack_col,
            core_cols,
            rich_cols,
            full_attack_values,
            full_core_signatures,
            full_behavior_z,
            full_adaec_order,
        )

        sensitivity_path = (
            results_dir
            / "adaec_v3_sensitivity.csv"
        )
        sensitivity.to_csv(
            sensitivity_path, index=False
        )
        print(
            sensitivity.to_string(index=False)
        )

    config = {
        "version": "A-DAEC V3",
        "dataset": DATASET_NAME,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "attack_column": attack_col,
        "binary_label_column": label_col,
        "sample_sizes": run_sizes,
        "sampling": (
            "deterministic proportional stratified "
            "sampling + deterministic shuffle"
        ),
        "random_seed": RANDOM_SEED,
        "max_group_size": MAX_GROUP_SIZE,
        "target_semantic_homogeneity": (
            TARGET_SEMANTIC_HOMOGENEITY
        ),
        "core_signature_columns": core_cols,
        "rich_semantic_order_columns": rich_cols,
        "independent_behavioral_columns": (
            behavior_cols
        ),
        "timing_repeats": TIMING_REPEATS,
        "verification_samples_per_method": (
            VERIFICATION_SAMPLES
        ),
        "sensitivity_thresholds": (
            SENSITIVITY_THRESHOLDS
        ),
        "cryptographic_hash": "SHA-256",
        "merkle_tree": True,
        "mining_or_pow": False,
        "onchain_record_bytes_assumption": (
            ONCHAIN_RECORD_BYTES
        ),
        "optimization_statement": (
            "Minimize blockchain commitments subject "
            "to 100% attack homogeneity, semantic "
            "homogeneity >= target, 100% audit coverage, "
            "100% verification success, 100% tamper "
            "detection, and group size <= max_group_size."
        ),
        "important_methodological_note": (
            "Independent Behavioral Cohesion Score "
            "(IBCS) uses numerical behavioral features "
            "that are not used by A-DAEC grouping."
        ),
    }

    config_path = (
        results_dir / "experiment_config_v3.json"
    )
    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 80)
    print(" FINAL RESULTS")
    print("=" * 80)

    summary_cols = [
        "Events",
        "Method",
        "Transactions",
        "TRR_pct",
        "Constraint_Feasible",
        "Attack_Homogeneity_pct",
        "Semantic_Homogeneity_pct",
        "IBCS_pct",
        "Algorithm_Time_Mean_s",
        "Verification_Time_ms",
        "Tamper_Detection_pct",
    ]

    print(
        results_df[summary_cols].to_string(index=False)
    )

    print("\nSaved files:")
    print(f"  {results_path}")
    if sensitivity_path:
        print(f"  {sensitivity_path}")
    print(f"  {config_path}")
    print(
        "  "
        + str(
            results_dir
            / "dataset_attack_distribution_v3.csv"
        )
    )
    print(
        "  "
        + str(
            results_dir
            / "sample_attack_distributions_v3.csv"
        )
    )

    print(
        "\nA-DAEC V3 completed successfully."
    )
    input("\nPress Enter to finish...")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print("\n" + "=" * 80)
        print("UNEXPECTED ERROR")
        print("=" * 80)
        print(type(exc).__name__ + ":", exc)
        print(
            "\nCopy the complete error and send it back."
        )
        input("\nPress Enter to finish...")
        exit_code = 1

    sys.exit(exit_code)
