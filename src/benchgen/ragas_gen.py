"""RAGAS question generation, as described in the RAGAs paper (Es et al., EACL 2024).

The paper builds its WikiEval dataset with two prompts, used here unchanged:
  1. formulate ONE question from a context, under six rules;
  2. answer that question using only the context.
There is no filtering step in the paper. (The newer testset generator inside the
`ragas` software library is a different, knowledge-graph-based method that is
NOT described in the paper, so it is not used.)
The paper used ChatGPT; we use DeepSeek.
"""
from src.benchgen.common import clean_question

SYSTEM = "You are a helpful assistant."

QUESTION_PROMPT = (
    "Your task is to formulate a question from given context satisfying the rules given below:\n"
    "1. The question should be fully answered from the given context.\n"
    "2. The question should be framed from a part that contains non-trivial information.\n"
    "3. The answer should not contain any links.\n"
    "4. The question should be of moderate difficulty.\n"
    "5. The question must be reasonable and must be understood and responded to by humans.\n"
    "6. Do not use phrases that 'provided context', etc in the question\n"
    "context: "
)

ANSWER_PROMPT = "Answer the question using the information from the given context.\n"


class RagasGenerator:
    name = "ragas"
    calls_per_attempt = 2
    overhead_tokens = 180

    def __init__(self, ctx):
        self.chat = ctx.chat

    def attempt(self, passage, rng):
        question = clean_question(self.chat(SYSTEM, QUESTION_PROMPT + passage["text"]))
        if not question:
            return {"rejected": "empty_question"}
        answer = self.chat(
            SYSTEM,
            f"{ANSWER_PROMPT}question: {question}\ncontext: {passage['text']}",
        )
        if not answer:
            return {"rejected": "empty_answer"}
        return {
            "question": question,
            "answer": answer,
            "passage_ids": [passage["id"]],
            "contexts": [passage["text"]],
            "meta": {},
        }
