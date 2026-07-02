import json

from dotenv import load_dotenv
load_dotenv()

from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import ResponseRelevancy

from rag_graph import rag_app, RAGState, llm, embeddings

# 1. Load questions
questions = json.load(open("eval.json"))

# 2. Run each question through the RAG pipeline
rows = []
for q in questions:
    result = rag_app.invoke(RAGState(question=q["question"]))
    rows.append({
        "user_input": q["question"],
        "response": result["answer"],
    })

# 3. Build the RAGAS dataset
dataset = EvaluationDataset.from_list(rows)

# 4. Evaluate
result = evaluate(
    dataset=dataset,
    metrics=[ResponseRelevancy(strictness=1)],
    llm=LangchainLLMWrapper(llm),
    embeddings=LangchainEmbeddingsWrapper(embeddings),
)

print(result)