import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.generate_answer import get_rag_response


def run_evaluation():

    df = pd.read_csv("evaluation/dataset.csv")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for _, row in df.iterrows():

        question = row["question"]
        ground_truth = row["ground_truth"]

        response = get_rag_response(question)

        questions.append(question)
        answers.append(response["answer"])
        contexts.append(response["contexts"])
        ground_truths.append(ground_truth)

    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    })

    print("Dataset Columns:")
    print(dataset.column_names)

    print("\nDataset Sample:")
    print(dataset[0])

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    result_df = result.to_pandas()

    result_df.to_csv(
        "evaluation/results.csv",
        index=False,
    )

    print(result_df)
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    run_evaluation()