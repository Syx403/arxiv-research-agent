from __future__ import annotations

import asyncio
import json

import pytest

from src.core.types import ChatResponse
from src.retrieval.index.types import Hit
from src.retrieval.self_rag import relevance_judge


@pytest.mark.asyncio
async def test_judge_batch_reduces_requests_and_preserves_identity_order(monkeypatch) -> None:
    active = 0
    max_active = 0
    calls = 0

    class Client:
        async def chat(self, messages, **kwargs):
            nonlocal active, max_active, calls
            active += 1
            calls += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            ids = [item["chunk_id"] for item in json.loads(messages[1].content)["excerpts"]]
            return ChatResponse(content=json.dumps({"results": [
                {"chunk_id": i, "role": "direct" if i % 2 == 0 else "irrelevant", "sentence_ids": [0] if i % 2 == 0 else [], "rationale": str(i)} for i in reversed(ids)
            ]}), model="fake", finish_reason="stop")

    monkeypatch.setattr(relevance_judge, "get_chat_client", lambda name: Client())
    hits = [
        Hit(chunk_id=idx, paper_id="p", section="S", text="text long enough", score=1.0)
        for idx in range(20)
    ]

    verdicts = await relevance_judge.judge_batch("subq", hits)

    assert len(verdicts) == 20
    assert calls == 4  # Four small batches, not twenty per-chunk requests.
    assert max_active <= 2
    assert [v.rationale for v in verdicts] == [str(i) for i in range(20)]
    assert [v.relevant for v in verdicts] == [i % 2 == 0 for i in range(20)]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"results": []},
    {"results": [{"chunk_id": 99, "role": "direct", "sentence_ids": [0], "rationale": "wrong identity"}]},
    {"results": [{"chunk_id": "1", "role": "direct", "sentence_ids": [0], "rationale": "wrong type"}]},
    {"results": [{"chunk_id": 1, "role": "direct", "sentence_ids": [0], "rationale": "duplicate"}] * 2},
])
async def test_invalid_batch_never_approves_unmapped_evidence(monkeypatch, payload):
    calls = []
    class Client:
        async def chat(self, messages, **kwargs):
            calls.append(messages)
            return ChatResponse(content=json.dumps(payload), model="fake", finish_reason="stop")
    monkeypatch.setattr(relevance_judge, "get_chat_client", lambda name: Client())
    verdicts = await relevance_judge.judge_batch("question", [Hit(chunk_id=1, paper_id="p", section="", text="evidence", score=1)])
    assert len(calls) == 2
    assert verdicts[0].relevant is None
    assert verdicts[0].status == "error"


@pytest.mark.asyncio
async def test_judge_relevance_repairs_invalid_json(monkeypatch) -> None:
    client = _RepairingChatClient()
    monkeypatch.setattr(relevance_judge, "get_chat_client", lambda name: client)

    verdict = await relevance_judge.judge_relevance(
        "What is ReAct?",
        Hit(chunk_id=1, paper_id="p", section="S", text="ReAct text", score=1.0),
    )

    assert verdict.relevant is True
    assert client.calls == 2


@pytest.mark.asyncio
async def test_judge_relevance_drops_hit_when_repair_is_invalid(monkeypatch) -> None:
    client = _BrokenRepairChatClient()
    monkeypatch.setattr(relevance_judge, "get_chat_client", lambda name: client)

    verdict = await relevance_judge.judge_relevance(
        "What is ReAct?",
        Hit(chunk_id=1, paper_id="p", section="S", text="ReAct text", score=1.0),
    )

    assert verdict.relevant is None
    assert verdict.error_code == "output_truncated"


class _RepairingChatClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *args, **kwargs) -> ChatResponse:
        self.calls += 1
        if self.calls == 1:
            return ChatResponse(content='{"relevant": true, "rationale": "truncated', model="fake", finish_reason="length")
        return ChatResponse(content='{"results":[{"chunk_id":1,"role":"direct","sentence_ids":[0],"rationale":"fixed"}]}', model="fake", finish_reason="stop")


class _BrokenRepairChatClient:
    async def chat(self, *args, **kwargs) -> ChatResponse:
        return ChatResponse(content="", model="fake", finish_reason="length")


@pytest.mark.asyncio
async def test_partial_batch_retries_only_bad_item_and_keeps_good_items(monkeypatch, tmp_path):
    from src.core.trace import LocalTrace
    calls = []
    class Client:
        async def chat(self, messages, **kwargs):
            ids = [x['chunk_id'] for x in json.loads(messages[1].content)['excerpts']]
            calls.append(ids)
            return ChatResponse(content=json.dumps({'results': [
                {'chunk_id':i, 'role':'direct', 'sentence_ids':['0'] if i == 2 and len(calls) == 1 else [0], 'rationale':'source explains mechanism'}
                for i in ids
            ]}), model='fake', finish_reason='stop')
    monkeypatch.setattr(relevance_judge, 'get_chat_client', lambda _: Client())
    path = tmp_path / 'trace.jsonl'
    with LocalTrace(path,run_id='partial',mode='test').activate():
        verdicts = await relevance_judge.judge_batch('mechanism', [Hit(i,'p','Methods','The mechanism is explained.',1) for i in [1,2,3]])
    assert calls == [[1,2,3],[2]]
    assert all(v.relevant is True and v.status == 'ok' for v in verdicts)
    events = [json.loads(x) for x in path.read_text().splitlines()]
    assert events[0]['content_length'] > 0
    assert events[0]['response_content']
    assert events[0]['errors'][0]['details'][0]['loc'] == ['sentence_ids',0]


@pytest.mark.asyncio
async def test_truncated_batch_is_split_for_bounded_recovery(monkeypatch):
    calls=[]
    thinking_modes=[]
    class Client:
        async def chat(self,messages,**kwargs):
            ids=[x['chunk_id'] for x in json.loads(messages[1].content)['excerpts']]
            calls.append(ids)
            thinking_modes.append(kwargs.get('thinking'))
            if len(ids)>2:
                return ChatResponse(content='',model='fake',finish_reason='length')
            return ChatResponse(content=json.dumps({'results':[
                {'chunk_id':i,'role':'direct','sentence_ids':[0],'rationale':'mechanism'} for i in ids
            ]}),model='fake',finish_reason='stop')
    monkeypatch.setattr(relevance_judge,'get_chat_client',lambda _:Client())
    verdicts=await relevance_judge.judge_batch('mechanism',[Hit(i,'p','Method','Valid source evidence.',1) for i in range(5)])
    assert calls[0] == list(range(5))
    assert sorted(map(len,calls[1:])) == [1,2,2]
    assert thinking_modes == [False, False, False, False]
    assert all(v.relevant for v in verdicts)


@pytest.mark.asyncio
async def test_extra_envelope_metadata_does_not_discard_valid_judgments(monkeypatch):
    calls = []

    class Client:
        async def chat(self, messages, **kwargs):
            calls.append(messages)
            return ChatResponse(content=json.dumps({
                'type': 'json_object',
                'results': [{'chunk_id': 1, 'role': 'direct', 'sentence_ids': [0], 'rationale': 'Mechanism evidence'}],
            }), model='fake', finish_reason='stop')

    monkeypatch.setattr(relevance_judge, 'get_chat_client', lambda _: Client())
    verdict = await relevance_judge.judge_relevance('mechanism', Hit(1, 'p', 'Method', 'A valid original sentence.', 1))
    assert len(calls) == 1
    assert verdict.status == 'ok' and verdict.relevant is True
    assert verdict.sentence_ids == [0]


@pytest.mark.asyncio
async def test_bibliography_cannot_become_mechanism_evidence(monkeypatch):
    monkeypatch.setattr(relevance_judge,'get_chat_client',lambda _:pytest.fail('No model call for a bibliography'))
    result=await relevance_judge.judge_batch('What is ReAct?', [Hit(1,'p','References','Yao et al. ReAct. 2023.',1)])
    assert result[0].status == 'ok'
    assert result[0].relevant is False


@pytest.mark.asyncio
@pytest.mark.parametrize('ids', [[99],[-1],[0,0],[True],['0'],[]])
async def test_invalid_selected_sentence_never_becomes_evidence(monkeypatch,ids):
    class Client:
        async def chat(self,*args,**kwargs):
            return ChatResponse(content=json.dumps({'results':[{'chunk_id':1,'role':'direct','sentence_ids':ids,'rationale':'claim'}]}),model='fake',finish_reason='stop')
    monkeypatch.setattr(relevance_judge,'get_chat_client',lambda _:Client())
    verdict=await relevance_judge.judge_relevance('question',Hit(1,'p','Method','One valid sentence.',1))
    assert verdict.relevant is None
    assert verdict.status == 'error'
