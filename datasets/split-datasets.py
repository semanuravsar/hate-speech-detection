import pandas as pd
from sklearn.model_selection import train_test_split

# Split latent hatred dataset
df_hatred = pd.read_csv("datasets/latent_hatred_3class_sample.csv")
train_h, test_h = train_test_split(df_hatred, test_size=0.2, random_state=42, stratify=df_hatred["label_id"])
train_h.to_csv("datasets/latent_hatred_3class_sample_train.csv", index=False)
test_h.to_csv("datasets/latent_hatred_3class_sample_test.csv", index=False)

# Split stereoset dataset
df_stereo = pd.read_csv("datasets/stereoset.csv")
train_s, test_s = train_test_split(df_stereo, test_size=0.2, random_state=42, stratify=df_stereo["label"])
train_s.to_csv("datasets/stereoset_train.csv", index=False)
test_s.to_csv("datasets/stereoset_test.csv", index=False)

# Split isarcasm dataset
df_sarcasm = pd.read_csv("datasets/isarcasm_sample.csv")
train_sarc, test_sarc = train_test_split(df_sarcasm, test_size=0.2, random_state=42, stratify=df_sarcasm["label"])
train_sarc.to_csv("datasets/isarcasm_sample_train.csv", index=False)
test_sarc.to_csv("datasets/isarcasm_sample_test.csv", index=False)

print("✅ Datasets successfully split into train and test.")
