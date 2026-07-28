"""
ml/leakage_demo.py
===================================================================
Proof that the validation method matters more than the model.

Runs four validation schemes on the SAME synthetic random walk — data
whose true predictable edge is exactly zero by construction — and shows
that the textbook sklearn approach reports a POSITIVE edge on pure noise.

Run it whenever you doubt whether the purging in validation.py is worth
the complexity:

    python3 ml/leakage_demo.py
===================================================================
"""
import sys, os, numpy as np
sys.path.insert(0, 'ml')
from features import build_features, build_labels, FEATURE_NAMES
from validation import purged_walk_forward, classification_metrics, aggregate_folds
import data as dm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Pure random walk: TRUE EDGE IS EXACTLY ZERO BY CONSTRUCTION
candles = dm.synthetic_random_walk(n=6000, drift=0.00004, seed=42)
X, valid = build_features(candles)
tb_label, touch_idx, realised_return, labelable = build_labels(candles['close'], vertical=24)
# Triple-barrier timeouts (label==0) are ambiguous and dropped, same as
# the main pipeline's binary training view — see ml/train.py build_dataset().
keep = valid & labelable & (tb_label != 0)
X, y = X[keep], (tb_label[keep] == 1).astype(int)
base = max(y.mean(), 1-y.mean())

print("="*70)
print("SAME DATA (random walk, true edge = 0), THREE VALIDATION METHODS")
print("="*70)
print(f"base rate (majority class) = {base*100:.2f}%   <- the score to beat\n")

pipe = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced'))

# METHOD 1: the textbook mistake — random shuffled K-fold
sc = cross_val_score(pipe(), X, y, cv=KFold(5, shuffle=True, random_state=0), scoring='accuracy')
print(f"1. Shuffled 5-fold KFold      acc {sc.mean()*100:5.2f}%   edge {(sc.mean()-base)*100:+6.2f}pp   <- LEAKY")

# METHOD 2: random single split
Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
m = pipe().fit(Xtr,ytr); a2 = m.score(Xte,yte)
print(f"2. Random train_test_split    acc {a2*100:5.2f}%   edge {(a2-base)*100:+6.2f}pp   <- LEAKY")

# METHOD 3: chronological split, NO purge (better, still leaks via overlapping labels)
cut = int(len(X)*0.8)
m = pipe().fit(X[:cut], y[:cut]); a3 = m.score(X[cut:], y[cut:])
b3 = max(y[cut:].mean(), 1-y[cut:].mean())
print(f"3. Chronological, no purge    acc {a3*100:5.2f}%   edge {(a3-b3)*100:+6.2f}pp   <- partial fix")

# METHOD 4: our harness
folds=[]
for tr,te in purged_walk_forward(len(X), n_splits=5, horizon=24, embargo=6):
    mm = pipe().fit(X[tr], y[tr])
    p = mm.predict_proba(X[te])[:,1]
    folds.append(classification_metrics(y[te], (p>=.5).astype(int), p))
agg = aggregate_folds(folds)
print(f"4. Purged walk-forward        acc {agg['accuracy_mean']*100:5.2f}%   edge {agg['edge_mean']*100:+6.2f}pp   <- CORRECT")
print()
print(f"The leaky methods overstate the edge by "
      f"{((sc.mean()-base) - agg['edge_mean'])*100:.1f} percentage points")
print("on data that contains no predictable structure whatsoever.")
print("="*70)
