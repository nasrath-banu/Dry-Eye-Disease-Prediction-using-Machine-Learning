# feature_importance_analysis.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import joblib
import os

DATA_PATH = "Dataset.xlsx"   # place file in same folder or change path

# --- Load ---
df = pd.read_excel(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

# --- Detect target ---
target = None
for c in df.columns:
    if 'dry' in c.lower():
        target = c
        break
if target is None:
    target = df.columns[-1]
print("Detected target column:", target)

# --- Feature engineering ---
# Split Blood pressure
bp_col = next((c for c in df.columns if 'blood' in c.lower()), None)
if bp_col:
    syst, dias = [], []
    for v in df[bp_col].astype(str):
        if "/" in v:
            try:
                s,d = v.split("/")
                syst.append(float(s)); dias.append(float(d))
            except:
                syst.append(np.nan); dias.append(np.nan)
        else:
            # simple fallback
            parts = [p for p in v.replace(","," ").split() if p.replace(".","",1).isdigit()]
            syst.append(float(parts[0]) if len(parts)>=1 else np.nan)
            dias.append(float(parts[1]) if len(parts)>=2 else np.nan)
    df["Systolic_BP"] = syst
    df["Diastolic_BP"] = dias
    df.drop(columns=[bp_col], inplace=True)
    print("Split BP into Systolic_BP and Diastolic_BP")

# --- Map categorical Y/N and Gender ---
yn_map = {"Y":1,"N":0,"Yes":1,"No":0,"yes":1,"no":0,"y":1,"n":0,"TRUE":1,"FALSE":0,"True":1,"False":0}
gender_map = {"Male":1,"M":1,"male":1,"Female":0,"F":0,"female":0}

for col in df.select_dtypes(include='object').columns:
    vals = df[col].dropna().unique()
    vals_lower = [str(v).strip().lower() for v in vals]
    if any(v in ("male","female","m","f") for v in vals_lower):
        df[col] = df[col].map(gender_map)
    elif len(vals) <= 6:
        df[col] = df[col].map(yn_map)
    else:
        # try numeric conversion else factorize
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            df[col], _ = pd.factorize(df[col])

# --- Fill numeric missing with median ---
for c in df.select_dtypes(include=[np.number]).columns:
    df[c].fillna(df[c].median(), inplace=True)

# Ensure target numeric 0/1
if df[target].dtype == object:
    df[target] = df[target].map(yn_map)
df[target] = df[target].fillna(0).astype(int)

# --- Clip to realistic ranges (optional but recommended) ---
ranges = {
    "Age": (18,45),
    "Sleep duration": (4,10),
    "Sleep quality": (1,5),
    "Stress level": (1,5),
    "Heart rate": (60,100),
    "Daily steps": (1000,20000),
    "Physical activity": (0,180),
    "Height": (150,200),
    "Weight": (50,100),
    "Average screen time": (1,10)
}
for col,(low,high) in ranges.items():
    if col in df.columns:
        df[col] = df[col].clip(lower=low, upper=high)

# --- Prepare X, y ---
X = df.drop(columns=[target])
y = df[target]

print("Total samples:", len(df), "Features:", X.shape[1])
print("Target distribution:\n", y.value_counts())

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

# --- Balance training set (random oversample) ---
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
print("Balanced training counts:", y_train_bal.value_counts().to_dict())

# --- Train RandomForest ---
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train_bal, y_train_bal)

# RF importances
rf_imp = pd.Series(rf.feature_importances_, index=X_train_bal.columns).sort_values(ascending=False)

# Permutation importance on unseen test set
perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)

# Univariate ANOVA F-scores on balanced training set
skb = SelectKBest(score_func=f_classif, k="all").fit(X_train_bal, y_train_bal)
f_scores = pd.Series(skb.scores_, index=X_train_bal.columns).sort_values(ascending=False)

# Mutual information
mi = mutual_info_classif(X_train_bal, y_train_bal, random_state=42)
mi_scores = pd.Series(mi, index=X_train_bal.columns).sort_values(ascending=False)

# --- Combine normalized scores ---
def normalize(s):
    s = s.copy().fillna(0)
    if s.max() == s.min():
        return s*0.0
    return (s - s.min()) / (s.max() - s.min())

comb = pd.DataFrame({
    "rf_imp": normalize(rf_imp.reindex(X.columns)),
    "perm_imp": normalize(perm_imp.reindex(X.columns)),
    "f_score": normalize(f_scores.reindex(X.columns)),
    "mi_score": normalize(mi_scores.reindex(X.columns))
}, index=X.columns)
comb["mean_score"] = comb.mean(axis=1)
comb = comb.sort_values("mean_score", ascending=False)

# Save outputs
out_dir = "feature_analysis_outputs"
os.makedirs(out_dir, exist_ok=True)
comb.to_csv(os.path.join(out_dir, "combined_feature_ranking.csv"))
rf_imp.to_csv(os.path.join(out_dir, "rf_importances.csv"))
perm_imp.to_csv(os.path.join(out_dir, "perm_importances.csv"))
f_scores.to_csv(os.path.join(out_dir, "f_scores.csv"))
mi_scores.to_csv(os.path.join(out_dir, "mi_scores.csv"))

# Plot top features (matplotlib, single plot)
topk = 12
top = comb.head(topk)
plt.figure(figsize=(10,6))
plt.bar(range(len(top)), top['mean_score'])
plt.xticks(range(len(top)), top.index, rotation=45, ha="right")
plt.ylabel("Normalized importance (mean)")
plt.title("Top features for Dry Eye Prediction (combined score)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "top_features_combined.png"))
print("Saved outputs in folder:", out_dir)

# Print top 12 features
print("\nTop features (combined ranking):")
print(top['mean_score'])
