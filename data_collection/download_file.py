import pandas as pd
from datasets import load_dataset

# 1. Load the native dataset
print("Downloading HuggingFaceH4/MATH-500...")
dataset = load_dataset("HuggingFaceH4/MATH-500")

# 2. Convert the 'test' split to a pandas DataFrame
df = dataset["test"].to_pandas()

# 3. Apply explicit Schema Mapping changes
# Renaming columns: problem -> question, solution -> cot_solution (optional)
df = df.rename(columns={
    "problem": "question",
    "answer": "answer"  # Already matches, but explicitly kept intact
})

# 4. Add an explicit ID field if it doesn't already exist in the schema
# This creates a zero-indexed row integer ID column named 'id'
if "id" not in df.columns:
    df.insert(0, "id", range(len(df)))

# 5. Save directly to your flat parquet target filename
output_filename = "math_500_test.parquet"
df.to_parquet(output_filename, index=False)

print(f"Success! Saved dataset with updated schema to '{output_filename}'")
print(f"Columns now present: {list(df.columns)}")
