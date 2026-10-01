from dataclasses import replace

import pytest

from src.core.types import ChatResponse, RouteDecision, SubQResult
from src.graph.nodes import retrieve, synthesize
from src.retrieval import evidence
from src.retrieval.index.types import Hit, SourceSpan
from src.retrieval.self_rag import sufficiency_check
from scripts.repair_corpus_sections import infer_section, normalize


def test_late_mechanism_survives_without_unrelated_prefix():
    text='Unrelated introductory background. ' * 40 + 'Reflexion stores self-reflections in long-term memory. The trajectory is short-term memory.'
    h=Hit(1,'reflexion','Method',text,1)
    spans=evidence.sentence_spans(text)
    ids=[i for i,s in enumerate(spans) if 'memory' in text[s.start:s.end]]
    chosen=evidence.select_sentences(h,ids,role='direct')
    assert min(s.start for s in chosen.evidence_spans)>800
    assert 'long-term memory' in evidence.evidence_text(chosen)
    assert 'Unrelated' not in evidence.evidence_text(chosen)
    assert chosen.text == text


def test_pdf_page_continuation_keeps_its_condition_in_the_selected_unit():
    import json
    from pathlib import Path
    fixture = json.loads(Path('tests/fixtures/retrieval_evidence.json').read_text())
    source = next(h for h in fixture['chunks'] if h['chunk_id'] == 1503)
    hit = Hit(**source, score=1)
    units = evidence.sentence_spans(hit.text)
    selected_id = next(i for i, span in enumerate(units)
                       if 'asynchronous occurrence' in hit.text[span.start:span.end])
    selected = evidence.select_sentences(hit, [selected_id], role='direct')
    excerpt = evidence.evidence_text(selected)
    assert 'In contrast, for decision making tasks' in excerpt
    assert 'asynchronous occurrence' in excerpt
    assert excerpt in hit.text  # A contiguous, unchanged source range.


@pytest.mark.asyncio
async def test_sufficiency_generation_and_verifier_receive_identical_selected_text(monkeypatch):
    text='Background. ' * 100 + 'The actor uses short-term trajectories and long-term reflections.'
    h=Hit(1,'arxiv:reflexion','Method',text,1,evidence_spans=(SourceSpan(1200,len(text)),),evidence_role='direct')
    prompts=[]
    class Client:
        async def chat(self,messages,**kwargs):
            prompts.append(messages[1].content)
            output='The actor uses both memories [arxiv:reflexion#1].' if 'Answer research' in messages[0].content else '{"sufficient":true,"missing_aspects":[]}'
            return ChatResponse(content=output,model='fake',finish_reason='stop')
    monkeypatch.setattr(sufficiency_check,'get_chat_client',lambda _:Client())
    monkeypatch.setattr(synthesize,'get_chat_client',lambda _:Client())
    await sufficiency_check.is_sufficient('What memory does the actor use?',[h])
    result=await synthesize.synthesize_node({'question':'What memory does the actor use?','evidence':[h]})
    checked=result['evidence_lookup'][1]
    assert all(checked.text in prompt for prompt in prompts)
    assert all('Background.' not in prompt for prompt in prompts)
    assert checked.source_spans == [(1200,len(text))]


def test_comparison_merges_different_spans_from_same_original_chunk():
    h=Hit(1,'p','Method','First mechanism. Second mechanism.',1,evidence_spans=(SourceSpan(0,16),))
    other=replace(h,evidence_spans=(SourceSpan(17,len(h.text)),))
    groups={i:SubQResult(subq_index=i,hits=[hit],sufficient=True,route_decision=RouteDecision()) for i,hit in enumerate([h,other])}
    result=retrieve._assemble_evidence(groups)
    selected=synthesize._select_synthesis_evidence(result,groups)
    assert len(selected)==1
    assert 'First mechanism.' in evidence.evidence_text(selected[0])
    assert 'Second mechanism.' in evidence.evidence_text(selected[0])


def test_context_budget_never_clips_a_selected_sentence():
    text='This sentence describes a memory mechanism in sufficient detail. ' * 90
    h=Hit(1,'p','Method',text,1)
    selected=evidence.select_sentences(h,list(range(len(evidence.sentence_spans(text)))),role='direct')
    assert evidence.token_count(evidence.evidence_text(selected))<=evidence.EVIDENCE_CHUNK_TOKENS
    assert all(h.text[s.start:s.end].endswith('.') for s in selected.evidence_spans)


def test_original_paper_precedes_secondary_background_without_padding():
    original=Hit(1,'react','Method','ReAct interleaves thoughts and actions.',.5,title='ReAct: Synergizing Reasoning and Acting',evidence_role='direct')
    background=Hit(2,'survey','Related work','ReAct is a prior method.',.99,title='A Survey of Agents',evidence_role='background')
    selected=evidence.pack_evidence([background,original],'What is ReAct?')
    assert [h.chunk_id for h in selected]==[1]
    assert len(evidence.pack_evidence([background,original],'How does later work compare with ReAct?'))==2
    assert not evidence.is_primary_source('How do agents react to feedback?', 'Reactors: Chemical processes')


def test_complementary_original_evidence_survives_background_role():
    title = 'Routing Method: Balanced Experts'
    mechanism = Hit(1, 'paper', 'Method', 'Heavy load reduces the bias.', .9,
                    title=title, evidence_role='direct')
    training = Hit(2, 'paper', 'Conclusion', 'No auxiliary loss gradients are introduced.', .8,
                   title=title, evidence_role='background')
    secondary = Hit(3, 'survey', 'Related work', 'This survey lists routing methods.', .99,
                    title='A Survey', evidence_role='background')
    selected = evidence.pack_evidence([mechanism, training, secondary],
                                     'How does Routing Method update bias and use gradients?')
    assert {h.chunk_id for h in selected} == {1, 2}


def test_bibliographies_remain_available_for_explicit_bibliography_request():
    h=Hit(1,'p','References','Yao et al. ReAct. 2023.',1)
    assert not evidence.is_reference_list(h,'Show the bibliography')
    assert evidence.is_reference_list(h,'Explain ReAct mechanism')


def test_section_repair_ignores_repeated_captions():
    repeated='Repeated figure caption ' * 15
    mechanism='The actual method description is unique and explains control flow. ' * 10
    sections=[('Introduction',normalize(repeated)),('Method',normalize(mechanism+repeated)),('Experiments',normalize(repeated))]
    assert infer_section(mechanism+repeated,sections)=='Method'


def test_selected_source_offsets_survive_checkpoint_serialization():
    from src.graph.builder import checkpoint_serializer
    h=Hit(1,'p','Methods','First. Second.',1,title='Paper',evidence_spans=(SourceSpan(7,14),),evidence_role='direct')
    serializer=checkpoint_serializer()
    restored=serializer.loads_typed(serializer.dumps_typed({'evidence':[h]}))
    assert restored['evidence'][0] == h
    assert evidence.evidence_text(restored['evidence'][0]) == 'Second.'


@pytest.mark.asyncio
async def test_failed_judgment_is_reported_as_unknown_not_missing_sources():
    from src.graph.nodes.finalize import finalize_node
    result=SubQResult(subq_index=0,sufficient=False,route_decision=RouteDecision(),
                      stop_reason='relevance_judgment_failed',judgment_failures=[1],
                      unassessed_hits=[Hit(1,'p','Method','A candidate source exists.',1)])
    final=await finalize_node({'answer':'','subq_results':{0:result},'evidence':[]})
    assert final['status']=='incomplete'
    assert final['stop_reason']=='evidence_judgment_failed'
    assert 'does not mean the sources are irrelevant or absent' in final['answer']
    assert 'retry the failed evidence assessment' in final['answer']


def test_diagnostic_fixture_has_valid_immutable_gold_spans():
    import json
    from pathlib import Path
    fixture=json.loads(Path('tests/fixtures/retrieval_evidence.json').read_text())
    chunks={c['chunk_id']:c for c in fixture['chunks']}
    for case in fixture['cases']:
        assert set(case['relevant_ids']) <= set(case['candidate_ids'])
        for span in case['required_spans']:
            assert chunks[span['chunk_id']]['text'][span['start']:span['end']] == span['text']
