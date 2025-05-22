import pandas as pd

df = pd.read_csv("datasets/latent_hatred_3class.csv")
sample = df.sample(n=100, random_state=42)
sample.to_csv("datasets/latent_hatred_3class_sample.csv", index=False)

df = pd.read_csv("datasets/stereoset.csv")
sample = df.sample(n=100, random_state=42)
sample.to_csv("datasets/stereoset_sample.csv", index=False)

df = pd.read_csv("datasets/isarcasm.csv")
sample = df.sample(n=100, random_state=42)
sample.to_csv("datasets/isarcasm_sample.csv", index=False)

