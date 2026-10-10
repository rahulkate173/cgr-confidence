import pandas as pd

# Load directly from the Hugging Face parquet cache branch
df = pd.read_parquet(
    "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test/0000.parquet"
)
df.to_parquet("math_500.parquet")