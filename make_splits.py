import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

SPREADSHEET = Path("/media/tfi38/Expansion/Malignant_Polyps/Slide_and_Clinical_Table_for_Datasets_28thJan2026.xlsx")
FEATURES_DIR = Path("/csse/users/tfi38/honours_project2/features/conch-7e9d3eb3")
SPLITS_DIR = Path("/csse/users/tfi38/honours_project2/splits")

import shutil

# Clean up old splits
if SPLITS_DIR.exists():
    shutil.rmtree(SPLITS_DIR)
    print("Cleared old splits")

# Load and combine all sheets
canterbury = pd.read_excel(SPREADSHEET, sheet_name='Canterbury_Cases')
waikato = pd.read_excel(SPREADSHEET, sheet_name='Waikato_Cases')
auckland = pd.read_excel(SPREADSHEET, sheet_name='Auckland_Cases')

# Add dataset prefix to patient IDs to avoid collisions
canterbury['Patient'] = 'C_' + canterbury['Patient'].astype(str)
waikato['Patient'] = 'W_' + waikato['Patient'].astype(str)
auckland['Patient'] = 'A_' + auckland['Patient'].astype(str)

df = pd.concat([canterbury, waikato, auckland], ignore_index=True)
df = df.rename(columns={'Patient': 'patient', 'StudyID/SlideID': 'slide_id', 'isLNM': 'label'})
df['label'] = df['label'].map({'Y': 1, 'N': 0})

# Find feature files
def find_h5(slide_id):
    matches = list(FEATURES_DIR.rglob(f"{slide_id}.h5"))
    return matches[0] if matches else None

df['h5_path'] = df['slide_id'].apply(find_h5)
df = df[df['h5_path'].notna()].reset_index(drop=True)
print(f"Working with {len(df)} slides")

# Create splits
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
splits = list(sgkf.split(df, df['label'], groups=df['patient']))
train_idx, test_idx = splits[0]

test_df = df.iloc[test_idx].copy()
train_df = df.iloc[train_idx].copy()

print(f"Test set: {len(test_df)} slides, {test_df['label'].sum()} positive")
print(f"Train pool: {len(train_df)} slides, {train_df['label'].sum()} positive")

# Save splits as symlinked flat folders
def save_split(split_df, split_dir):
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for _, row in split_df.iterrows():
        src = Path(row['h5_path'])
        label_str = "positive" if row['label'] == 1 else "negative"
        dst = split_dir / f"{src.stem}_{label_str}.h5"
        if not dst.exists():
            dst.symlink_to(src)
    print(f"Saved {len(split_df)} files to {split_dir}")

save_split(test_df, SPLITS_DIR / "test")

for fold, (t_idx, v_idx) in enumerate(sgkf.split(train_df, train_df['label'], groups=train_df['patient'])):
    fold_train = train_df.iloc[t_idx]
    fold_val = train_df.iloc[v_idx]
    save_split(fold_train, SPLITS_DIR / f"fold_{fold+1}" / "train")
    save_split(fold_val, SPLITS_DIR / f"fold_{fold+1}" / "val")
    print(f"Fold {fold+1}: train={len(fold_train)}, val={len(fold_val)}, val_pos={fold_val['label'].sum()}")
