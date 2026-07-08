# FGO Agentic RAG

An agentic Retrieval-Augmented Generation system over *Fate/Grand Order* game data — structured game-mechanic facts (skills, noble phantasm ranks, rarity) and unstructured lore (4,000+ servant profiles and story/quest scripts) in Chinese and Japanese. Built to explore, hands-on, the gap between "RAG demo that works on the happy path" and "RAG system that behaves correctly when retrieval is ambiguous, sources conflict, or a user asks something the corpus can't answer."

## Why this exists

Most RAG tutorials stop at "embed documents, retrieve top-k, stuff into a prompt." This project pushes past that into the failure modes that actually show up once a RAG system has to sit in a real conversation: retrieval returning documents that contradict each other, a cross-encoder reranker silently blowing up its own latency on long documents, a clarification loop that never converges, questions that need every occurrence of an entity rather than the top-k most similar. Each of those was found by dogfooding the system, diagnosed from first principles, and fixed — see [Engineering log](#engineering-log-what-actually-broke-and-how-it-was-fixed) below.

## Architecture

```
                      ┌─────────────────────────────────────────┐
                      │              Data pipeline               │
                      │  Atlas Academy API ──▶ servants.db (SQL) │
                      │  fgo.wiki scrape   ──▶ wiki_raw/*.json   │
                      │  Atlas quest scripts ▶ quest_raw/*.json  │
                      │       │                                  │
                      │       ▼                                  │
                      │  BM25 index + bge-m3 dense index (Qdrant)│
                      │  (long docs: LLM-summarized before embed)│
                      └─────────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────── agent/graph.py (outer) ─────────────────────────────┐
│  resolve_question ──▶ decompose ──▶ solve_subquestions ──▶ synthesize            │
│  (context memory,    (multi-hop      (per sub-question:      (combine sub-       │
│   ask if ambiguous)   split +        route to subgraph        answers into one   │
│                        query_type    OR exhaustive             final answer)     │
│                        routing)      keyword scan)                               │
└─────────────────────────────────────┬─────────────────────────────────────────────┘
                                       │  query_type="standard"
                                       ▼
                    ┌───────────── agent/subgraph.py (Self-RAG) ─────────────┐
                    │  route_question ──▶ structured_lookup / retrieve       │
                    │       │                       │                        │
                    │       ▼                       ▼                        │
                    │  servants.db            BM25 + bge-m3 + RRF fusion     │
                    │  (exact facts)           + bge-reranker-v2-m3          │
                    │                          + score-cliff cutoff          │
                    │                                │                        │
                    │                                ▼                        │
                    │                          check_conflict ──▶ generate   │
                    │                          (structured route      │      │
                    │                           only)                 ▼      │
                    │                                          grade_generation│
                    │                                     (hallucination +   │
                    │                                      answer-quality,   │
                    │                                      retry loops)      │
                    └──────────────────────────────────────────────────────┘
```

**Two retrieval paths**, chosen per sub-question:
- **Structured** (`servants.db`, SQLite) — exact game-mechanic facts: skill effects, noble phantasm rank/card type, rarity, acquisition method. No hallucination risk; it's a lookup, not a generation.
- **Vectorstore** — BM25 (keyword) + bge-m3 (dense) → Reciprocal Rank Fusion → bge-reranker-v2-m3 (cross-encoder) → a dynamic cutoff that keeps candidates up to the steepest score drop in the ranked list, not a fixed top-k (see engineering log).

**Self-correction** (Self-RAG-style, via structured LLM judgments rather than fine-tuned reflection tokens):
- `check_conflict` — before generating, checks whether retrieved documents actually disagree on the fact being asked (e.g. two servant variants with different noble phantasm ranks) and asks the user to disambiguate instead of averaging over a real distinction.
- `grade_generation` — checks the answer for hallucination (unsupported claims) and relevance, retrying generation or retrieval as needed (bounded retries).

**Conversation memory + clarification**: `resolve_question` rewrites pronoun/reference-dependent follow-ups ("她的宝具是什么") into self-contained questions using conversation history, or asks a clarifying question if history doesn't resolve it — bounded to one round before the system commits to a best-effort interpretation rather than looping.

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph (two nested `StateGraph`s: outer multi-hop planner, inner Self-RAG solver) |
| LLM | gpt-4o-mini via an OpenAI-compatible endpoint (`langchain-openai`) |
| Dense embeddings | `BAAI/bge-m3` |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder) |
| Keyword search | `rank_bm25` + `jieba` (Chinese segmentation) |
| Vector store | Qdrant, local/embedded mode (no server) |
| Structured store | SQLite |
| Frontend | Streamlit (multi-turn chat) |
| Eval | Custom Recall@5 harness + RAGAS (faithfulness/relevancy/precision/recall) |

## Setup

```bash
pip install -e ".[retrieval,agent,eval,dev]"
cp .env.example .env   # fill in LLM_API_KEY at minimum
```

Build the corpus and indexes (resumable — re-running skips cached steps):

```bash
python scripts/update_all.py          # Atlas API -> servants.db, fgo.wiki scrape -> wiki_raw/
python scripts/fetch_quest_scripts.py # quest/story script text -> quest_raw/
python scripts/summarize_corpus.py    # LLM summaries for long records (used for embedding/rerank)
python scripts/build_bm25_index.py
python scripts/build_vector_index.py
```

Run it:

```bash
streamlit run app.py                  # chat UI
python -m agent.graph "阿尔托莉雅和贞德的宝具阶级哪个更高？"   # CLI, single query
```

## Eval results

25 hand-written questions ([`eval/questions.json`](eval/questions.json)) spanning single-servant facts, cross-servant comparisons, and JP-exclusive lore not yet in the CN client. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the full before/after numbers from the debugging process below.

| Metric | Result |
|---|---|
| Recall@5 (retrieval) | **96%** (24/25) |
| LLM calls per multi-hop query (typical) | 25 (down from 54 before removing per-document grading) |

## Engineering log: what actually broke, and how it was fixed

These were found by running real queries through the system, not by inspection — each is a concrete before/after with the observed failure. Full details in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

- **Reranker latency**: uncapped `max_length` let the cross-encoder tokenize candidates to 8192 tokens (256× the compute of a normal 512-token pass) whenever a candidate was a long story chunk — one query took 20+ minutes. Fixed with an explicit `max_length=512` cap, and long candidates get reranked against their LLM-generated summary instead of a truncated slice of raw text (recovers the recall the cap would otherwise cost).
- **Per-document grading was the dominant latency cost**: one LLM call per retrieved candidate to judge relevance (10+ calls for an ambiguous structured lookup) — removed entirely, replaced with a free (no LLM call) cutoff at the steepest score drop in the reranked list.
- **Silent top-5 truncation**: `retrieve()` requested only the top 5 reranked candidates, so the cliff-detection logic above could never see or recover a real hit ranked 6th+ — confirmed with a case where a servant's own profile page (0.895 rerank score) was excluded by a hard top-5 cutoff despite no real score cliff before it. Now requests the reranker's full candidate pool (free — it's already scored) and lets the cliff logic decide.
- **Infinite clarification loop**: open-ended narrative questions have no natural "specific enough" stopping point, so the clarification-detection logic could keep narrowing forever (observed: 8 rounds narrowing a relationship question with no answer ever generated). Fixed by scoping conflict-detection to the structured (fact) route only, and capping clarification rounds so the system commits to a best-effort answer after one round instead of asking indefinitely.
- **Top-k retrieval can't answer "list every X"**: semantic/keyword top-k search only surfaces documents matching a specific query's phrasing — two rephrasings of "list every quest Jason appears in" returned non-overlapping subsets (6 and 2 records) of what turned out to be 130 actual occurrences. Added a `query_type` classification at decomposition time that routes enumeration-style questions to an exhaustive keyword scan instead of similarity search.

## Known limitations

- Enumeration-style questions do an exact substring scan, not a semantic one — pronoun-only mentions of an entity won't be counted.
- The relevance cutoff (`MIN_KEPT_DOCUMENTS` floor, no upper cap) can occasionally pull in more candidates than needed when the score curve declines gradually rather than cliff-like; not currently bounded.
- No automated test suite yet (see repo TODO).

## Project structure

```
agent/            LangGraph outer graph + Self-RAG subgraph, LLM client, schemas
retrieval.py      Hybrid retriever (BM25 + dense + RRF + rerank)
scripts/          Data pipeline (fetch, scrape, build indexes) + eval harnesses
app.py            Streamlit chat frontend
eval/             Hand-written eval question set + recall results
docs/BENCHMARKS.md  Detailed before/after metrics from the debugging process
```
