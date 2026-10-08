"""Read then answer: one question over given papers, without the conversation graph. S2 and the
live checks use it; the top-level graph (app.py) runs the same two subgraphs behind understand."""

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
