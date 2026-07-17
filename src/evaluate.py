import time
import pandas as pd

from rag_chatbot import answer_question


# Load the Golden Dataset
golden_df = pd.read_csv("golden_dataset/golden_questions.csv")

print(f"\nLoaded {len(golden_df)} evaluation questions.\n")

# Loop through every question
for index, row in golden_df.iterrows():

    question = row["question"]

    print("=" * 80)
    print(f"Question {index + 1}/{len(golden_df)}")
    print(question)

    while True:
        try:
            # Measure response time
            start_time = time.time()

            rag_response, sources = answer_question(question)

            end_time = time.time()

            response_time = end_time - start_time

            print("Completed!")
            print(f"Response Time: {response_time:.2f} seconds")

            break

        except Exception as e:

            print(f"\nError: {e}")

            # Handle Gemini rate limits
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("Rate limit reached. Waiting 15 seconds before retrying...\n")
                time.sleep(15)
            else:
                print("Unexpected error. Skipping this question.\n")
                break

print("\nEvaluation Completed Successfully!")