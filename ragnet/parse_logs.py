import re
import sys
from collections import defaultdict

BATCH_RE = re.compile(r"Question ID:\s*(WebQTest-\d+)")
METRIC_RE = re.compile(r"TP:\s*(\d+),\s*FP:\s*(\d+),\s*FN:\s*(\d+)")
NEW_RUN_MARKER = "NEW RUN STARTING"
FP_OUTLIER_THRESHOLD = 40000


def safe_div(a, b):
    return a / b if b > 0 else 0.0


def compute_metrics(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def main(logfile):
    with open(logfile, "r", encoding="utf-8") as f:
        text = f.read()

    # -----------------------------------------
    # Only keep content after LAST NEW RUN
    # -----------------------------------------
    if NEW_RUN_MARKER in text:
        text = text.split(NEW_RUN_MARKER)[-1]

    lines = text.splitlines()

    batch_latest = {}   # batch -> (tp, fp, fn)

    current_batch = None

    for line in lines:
        bmatch = BATCH_RE.search(line)
        if bmatch:
            current_batch = bmatch.group(1)

        mmatch = METRIC_RE.search(line)
        if mmatch and current_batch:
            tp, fp, fn = map(int, mmatch.groups())
            # overwrite → ensures we keep cumulative final value
            batch_latest[current_batch] = (tp, fp, fn)

    # -----------------------------------------
    # Remove FP outlier batches
    # -----------------------------------------
    filtered = {}
    for b, (tp, fp, fn) in batch_latest.items():
        if fp < FP_OUTLIER_THRESHOLD:
            filtered[b] = (tp, fp, fn)

    # -----------------------------------------
    # Per-batch metrics
    # -----------------------------------------
    print("\nPer-batch metrics (after outlier removal):\n")
    macro_p = []
    macro_r = []
    macro_f = []

    total_tp = total_fp = total_fn = 0

    for batch in sorted(filtered.keys(), key=lambda x: int(x.split("-")[1])):
        tp, fp, fn = filtered[batch]
        p, r, f1 = compute_metrics(tp, fp, fn)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        macro_p.append(p)
        macro_r.append(r)
        macro_f.append(f1)

        print(f"{batch:12s}  TP={tp:6d} FP={fp:6d} FN={fn:6d} "
              f"P={p:.4f} R={r:.4f} F1={f1:.4f}")

    # -----------------------------------------
    # Macro averages
    # -----------------------------------------
    macroP = sum(macro_p) / len(macro_p)
    macroR = sum(macro_r) / len(macro_r)
    macroF = sum(macro_f) / len(macro_f)

    # -----------------------------------------
    # Micro averages
    # -----------------------------------------
    microP = safe_div(total_tp, total_tp + total_fp)
    microR = safe_div(total_tp, total_tp + total_fn)
    microF = safe_div(2 * microP * microR, microP + microR)

    print("\n================ SUMMARY ================\n")
    print(f"Macro Precision: {macroP:.4f}")
    print(f"Macro Recall:    {macroR:.4f}")
    print(f"Macro F1:        {macroF:.4f}\n")

    print(f"Micro Precision: {microP:.4f}")
    print(f"Micro Recall:    {microR:.4f}")
    print(f"Micro F1:        {microF:.4f}\n")

    print(f"Total TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Batches counted: {len(filtered)}")
    print(f"Batches skipped as FP outliers: {len(batch_latest) - len(filtered)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_metrics.py <logfile.txt>")
        sys.exit(1)

    main(sys.argv[1])

