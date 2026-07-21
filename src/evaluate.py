import time
import pandas as pd
from pathlib import Path

from rag_chatbot import answer_question

# -------------------------------------------------
# Paths
# -------------------------------------------------

GOLDEN_DATASET = Path("golden_dataset/golden_questions.csv")
OUTPUT_FILE = Path("golden_dataset/evaluation_results.csv")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

golden_df = pd.read_csv(GOLDEN_DATASET)

print(f"\nLoaded {len(golden_df)} evaluation questions.\n")

# -------------------------------------------------
# Resume if file already exists
# -------------------------------------------------

if OUTPUT_FILE.exists():
    results_df = pd.read_csv(OUTPUT_FILE)
    completed = len(results_df)
    print(f"Resuming from Question {completed + 1}\n")
else:
    results_df = pd.DataFrame(columns=[
        "Question",
        "Expected Topic",
        "Expected Document",
        "Difficulty",
        "Retrieved Documents",
        "Correct Retrieval",
        "Response Time (s)",
        "Generated Answer"
    ])
    completed = 0

# -------------------------------------------------
# Evaluation
# -------------------------------------------------

for index in range(completed, len(golden_df)):

    row = golden_df.iloc[index]

    question = row["question"]
    expected_topic = row["expected_topic"]
    expected_document = row["expected_document"]
    difficulty = row["difficulty"]

    print("=" * 80)
    print(f"Question {index + 1}/{len(golden_df)}")
    print(question)

    try:

        start = time.time()

        rag_response, sources = answer_question(question)

        end = time.time()

        response_time = round(end - start, 2)

        retrieved_docs = list(sources.keys())

        correct = any(
            expected_document.lower() in doc.lower()
            for doc in retrieved_docs
        )

        new_row = pd.DataFrame({
            "Question":[question],
            "Expected Topic":[expected_topic],
            "Expected Document":[expected_document],
            "Difficulty":[difficulty],
            "Retrieved Documents":[", ".join(retrieved_docs)],
            "Correct Retrieval":[correct],
            "Response Time (s)":[response_time],
            "Generated Answer":[rag_response.answer]
        })

        results_df = pd.concat([results_df, new_row], ignore_index=True)

        results_df.to_csv(OUTPUT_FILE, index=False)

        print(f"✅ Retrieved: {retrieved_docs}")
        print(f"✅ Correct: {correct}")
        print(f"⏱ {response_time:.2f}s")

    except Exception as e:

        print(e)

        if "GenerateRequestsPerDay" in str(e):
            print("\nDaily quota exhausted.")
            print("Progress saved.")
            break

        elif "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("Rate limit. Waiting 15 seconds...\n")
            time.sleep(15)
            continue

        else:
            print("Skipping...\n")
            continue

# -------------------------------------------------
# Summary
# -------------------------------------------------

print("\n" + "="*80)

completed = len(results_df)
correct = results_df["Correct Retrieval"].sum()

accuracy = (correct/completed)*100 if completed else 0

print("Evaluation Complete")
print(f"Questions Completed : {completed}")
print(f"Retrieval Accuracy  : {accuracy:.2f}%")
print(f"Average Response    : {results_df['Response Time (s)'].mean():.2f}s")

print(f"\nSaved to:\n{OUTPUT_FILE}")