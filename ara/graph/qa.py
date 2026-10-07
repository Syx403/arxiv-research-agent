"""Read then answer: one question over given papers. The top-level graph (M3) puts the same two
subgraphs behind understand and discover; until then this is the entry for evals and checks."""

from ara.graph import answer, read
from ara.graph.state import Answer, Context

READ = read.build()
ANSWER = answer.build()


async def answer_question(question: str, papers: list[str], context: Context) -> Answer:
    found = await READ.ainvoke({"question": question, "papers": papers}, context=context)
    result = await ANSWER.ainvoke(
        {
            "question": question,
            "evidence": found.get("evidence", []),
            "missing": found.get("missing", []),
        },
        context=context,
    )
    delivered: Answer = result["answer"]
    return delivered
