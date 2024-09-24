import pandas as pd
from typing import List

def load_documents(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    documents = [
        f"Question: {row['question']} Answer: {row['answer_solution']}"
        for _, row in df.iterrows()
    ]
    return documents
