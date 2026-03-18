"""
DATASET VALIDATION DIAGNOSTIC — Read-Only.
Validates shapes, dimensions, NaN/Inf, and numerical sanity.
Does NOT modify any data.
"""
import sys, os, torch, numpy as np, pandas as pd
sys.path.insert(0, ".")
from src.dataset import (
    generate_sequences_per_item,
    MultivariateSlidingWindowDataset,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
)

csv = "food_evaluation_dataset.csv"
print(f"Loading {csv}...")
df = pd.read_csv(csv)
n_items_total = df["item_id"].nunique()
print(f"Full dataset: {df.shape[0]:,} rows, {n_items_total:,} items")

# Sort (required)
df["day_num"] = df["d"].astype(str).str.replace("d_", "", regex=False).astype(int)
df = df.sort_values(["item_id", "day_num"]).reset_index(drop=True)

# Sample 50 items for fast validation
sample_items = sorted(df["item_id"].unique()[:50])
df_sample = df[df["item_id"].isin(sample_items)].copy()
print(f"Sampled: {df_sample.shape[0]:,} rows, {len(sample_items)} items")

N_FEATURES = len(FEATURE_COLUMNS)
BATCH_SIZE = 64

# Generate sequences
X, y = generate_sequences_per_item(df_sample, sequence_length=SEQUENCE_LENGTH)
dataset = MultivariateSlidingWindowDataset(X, y)

print()
print("=" * 60)
print("  DATASET VALIDATION REPORT")
print("=" * 60)

# STEP 1
sample_X, sample_y = dataset[0]
print(f"\n[1] SAMPLE SHAPE CHECK")
print(f"  sample_X.shape = {tuple(sample_X.shape)}")
print(f"  sample_y.shape = {tuple(sample_y.shape)}")
shape_x_ok = sample_X.shape == (SEQUENCE_LENGTH, N_FEATURES)
shape_y_ok = sample_y.shape == (1,)
print(f"  Expected X: ({SEQUENCE_LENGTH}, {N_FEATURES}) -> {'PASS' if shape_x_ok else 'FAIL'}")
print(f"  Expected y: (1,)              -> {'PASS' if shape_y_ok else 'FAIL'}")

# STEP 2
total = len(dataset)
print(f"\n[2] DATASET SIZE CHECK")
print(f"  Total sequences: {total:,}")
print(f"  {'PASS' if total >= 100 else 'WARNING: small dataset'}")

# STEP 3
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
batch_X, batch_y = next(iter(loader))
actual_bs = batch_X.shape[0]
print(f"\n[3] BATCH SHAPE CHECK")
print(f"  batch_X.shape = {tuple(batch_X.shape)}")
print(f"  batch_y.shape = {tuple(batch_y.shape)}")
batch_x_ok = batch_X.shape == (actual_bs, SEQUENCE_LENGTH, N_FEATURES)
batch_y_ok = batch_y.shape == (actual_bs, 1)
dim_order_ok = batch_X.shape[1] == SEQUENCE_LENGTH and batch_X.shape[2] == N_FEATURES
print(f"  Expected X: ({actual_bs}, {SEQUENCE_LENGTH}, {N_FEATURES}) -> {'PASS' if batch_x_ok else 'FAIL'}")
print(f"  Expected y: ({actual_bs}, 1)        -> {'PASS' if batch_y_ok else 'FAIL'}")
print(f"  Dim order (batch, seq, feat)  -> {'PASS' if dim_order_ok else 'FAIL: WRONG ORDER'}")

# STEP 4
print(f"\n[4] NUMERICAL SANITY CHECK")
print(f"  First sample X (day 1):  {batch_X[0, 0, :].numpy()}")
print(f"  First sample X (day 30): {batch_X[0, -1, :].numpy()}")
print(f"  First sample y: {batch_y[0].item():.4f}")
x_min, x_max = dataset.X.min().item(), dataset.X.max().item()
y_min, y_max = dataset.y.min().item(), dataset.y.max().item()
print(f"  X value range: [{x_min:.4f}, {x_max:.4f}]")
print(f"  y value range: [{y_min:.4f}, {y_max:.4f}]")
neg_sales_x = (dataset.X[:, :, 0] < 0).any().item()
neg_sales_y = (dataset.y < 0).any().item()
print(f"  Negative sales in X: {'YES - FAIL' if neg_sales_x else 'No - PASS'}")
print(f"  Negative sales in y: {'YES - FAIL' if neg_sales_y else 'No - PASS'}")

# STEP 5
full_nan_x = torch.isnan(dataset.X).any().item()
full_nan_y = torch.isnan(dataset.y).any().item()
full_inf_x = torch.isinf(dataset.X).any().item()
full_inf_y = torch.isinf(dataset.y).any().item()
print(f"\n[5] STRICT ERROR DETECTION")
print(f"  NaN in X: {full_nan_x}  |  NaN in y: {full_nan_y}")
print(f"  Inf in X: {full_inf_x}  |  Inf in y: {full_inf_y}")

# STEP 6
print(f"\n{'=' * 60}")
print(f"  FINAL VALIDATION REPORT")
print(f"{'=' * 60}")
print(f"  Sample X shape:   {tuple(sample_X.shape)}")
print(f"  Sample y shape:   {tuple(sample_y.shape)}")
print(f"  Total sequences:  {total:,}")
print(f"  Batch X shape:    {tuple(batch_X.shape)}")
print(f"  Batch y shape:    {tuple(batch_y.shape)}")
print(f"  Contains NaN:     {full_nan_x or full_nan_y}")
print(f"  Contains Inf:     {full_inf_x or full_inf_y}")
print()

errors = []
if not shape_x_ok:
    errors.append(f"Sample X shape mismatch: {tuple(sample_X.shape)}")
if not shape_y_ok:
    errors.append(f"Sample y shape mismatch: {tuple(sample_y.shape)}")
if not dim_order_ok:
    errors.append("Dimension order incorrect")
if full_nan_x or full_nan_y:
    errors.append("Contains NaN values")
if full_inf_x or full_inf_y:
    errors.append("Contains Inf values")
if total < 10:
    errors.append(f"Dataset too small: {total}")

if errors:
    print("  Validation verdict: INVALID")
    for e in errors:
        print(f"     -> {e}")
else:
    print("  Validation verdict: VALID")
    print("     All shapes correct, no NaN/Inf, dimension order (batch, seq, feat) confirmed.")
print()
