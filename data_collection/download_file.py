from datasets import load_dataset

# Load the dataset
dataset = load_dataset("HuggingFaceH4/MATH-500")

# Save to parquet format locally
dataset.save_to_disk("./math_500_parquet")
# Or save a specific split directly to a parquet file using pandas:
dataset["test"].to_pandas().to_parquet("math_500_test.parquet")