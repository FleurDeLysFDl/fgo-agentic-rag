# Benchmarks & engineering log

Concrete before/after numbers from debugging this system against real queries, not synthetic ones. Corpus: 4,033 records (391 servant profiles + 3,572 quest/story script chunks), Chinese/Japanese.

## Retrieval quality: Recall@5

Eval harness: [`scripts/eval_recall.py`](../scripts/eval_recall.py), 25 hand-written questions ([`eval/questions.json`](../eval/questions.json)) covering single-servant facts, cross-servant comparisons, and JP-exclusive lore not yet ported to the CN client. A hit = the expected source document appears in the reranked top 5.

| Stage | Recall@5 | Notes |
|---|---|---|
| Baseline (reranker `max_length` uncapped) | *unmeasurable* | 20+ min per question; killed before completion (see latency section) |
| `max_length=512` cap added | 92% (23/25) | fixed the latency, but truncation silently dropped one answer that was past the 512-token window |
| + summary-based reranking for long candidates | **96% (24/25)** | long candidates reranked against their LLM-generated summary (already used for dense embedding) instead of a truncated raw-text slice; recovered the case above |

The one remaining miss is a same-event/different-chapter disambiguation the reranker genuinely can't resolve from the query alone (not a pipeline defect).

## Reranker latency

`bge-reranker-v2-m3`'s `CrossEncoder` falls back to the tokenizer's `model_max_length` (8192 for this model) if `max_length` isn't set explicitly. Candidate chunks in this corpus run up to ~37K characters (full quest scripts), so uncapped reranking was tokenizing every candidate pair to 8192 tokens — cross-encoder attention cost scales quadratically with sequence length, so this is ~256× the compute of a normal 512-token pass.

| Config | Time for 2/25 eval questions |
|---|---|
| `max_length` uncapped | 20+ minutes (process killed) |
| `max_length=512` | a few seconds |

## Multi-hop query LLM call count

Query: *"阿尔托莉雅和贞德的宝具阶级哪个更高？"* (a 3-sub-question comparison after decomposition).

| Change | LLM calls | Wall time |
|---|---|---|
| Baseline (`grade_documents`: 1 LLM call per retrieved candidate) | 54 | 246.0s |
| `grade_documents` removed entirely (no relevance filter) | 25 | 249.4s* / 115.2s (clean run) |
| Replaced with free score-gap cutoff (`_select_by_score_gap`, no LLM call) | 25 | 121.0s |

\* One run hit an anomalous 125s single-call stall on the upstream LLM proxy, unrelated to the code change — confirmed by a clean rerun at 115.2s. This also surfaced a missing request timeout on the LLM client (fixed: 60s timeout + 2 retries, see `agent/llm.py`).

Removing per-document grading cut LLM calls by more than half without changing Recall@5, because the cross-encoder reranker's own score ordering is already a good relevance signal — the grading step was re-deriving information the pipeline already had.

## Silent top-5 retrieval cutoff

`retrieve()` requested only `top_k=5` from the reranked candidate list before applying the score-gap cutoff, so the cutoff logic could never see (or recover) a genuine hit ranked 6th or lower.

Concrete case — query *"美狄亚与伊阿宋之间的关系是什么？"*: the servant's own profile page (`美狄亚`) reranked at **0.8954, 6th place** — a 0.012 gap from 5th place (0.9076), nowhere near a real cliff — but was excluded purely by the top-5 slice. The real cliff in that candidate list was down at rank 13 (0.7211 → 0.5460, a 0.175 drop).

Fix: `retrieve()` now requests the reranker's full candidate pool (`FUSED_TOP_K=20`) — free, since the reranker already scores all 20 regardless of what slice is requested back — and lets the score-gap cutoff make the real decision. Confirmed fix: the same query now returns the profile page correctly (rank 5, 0.9491).

Trade-off: without an upper bound, a gradually-declining score curve (no sharp cliff) can keep most of the 20 candidates — observed one query keeping 17/20. Accepted as a known trade-off rather than adding an arbitrary ceiling back.

## Enumeration-style questions: top-k vs. exhaustive scan

Top-k similarity search fundamentally cannot answer "list every quest/chunk mentioning X" — it returns whichever handful of records best match *this specific query's phrasing*, not every record containing the entity.

Concrete case — *"列出伊阿宋出场过的所有剧情章节标题"*, tested with two different phrasings:

| Phrasing | Records returned | Overlap between the two |
|---|---|---|
| "伊阿宋的出场剧情是什么？" | 6 | 0 |
| "列出伊阿宋的出场章节标题。" | 2 | 0 |
| Exhaustive keyword scan (`enumerate_by_keyword`) | **130** | — |

Fix: `decompose()` now classifies each sub-question's `query_type` (`standard` vs `enumerate`) at planning time; `enumerate` routes to a full-corpus substring scan (cheap — under a second over ~4,000 records) instead of top-k retrieval, returning a complete source list rather than a narrative synthesis (no LLM call needed).

## Multi-turn memory & clarification

- **Pronoun resolution**: *"她的宝具是什么？"* with no prior context correctly triggers a clarifying question ("请问你指的是哪位从者？"); after the user specifies *"阿尔托莉雅·潘德拉贡(Lancer)"*, a follow-up *"她的宝具阶级是什么？"* correctly resolves "她" to that specific servant+class using conversation history, without re-asking.
- **Conflict detection over guessing**: *"贞德的宝具阶级是什么？"* (ambiguous — matches both the Archer and Ruler variants, different noble phantasm ranks) triggers *"你想问的是贞德（archer）的宝具阶级A+，还是贞德（ruler）的宝具阶级A？"* — naming the actual conflicting values — instead of blending both variants' data into one answer (an earlier version did exactly that, producing "C或EX，具体取决于形态" by mixing two different servants' facts into one sentence).
- **Bounded clarification**: capping `MAX_CLARIFICATION_ROUNDS=1` was necessary because the LLM does not reliably recognize "the user's short reply already answers my own clarifying question" — observed a case where the model re-asked an almost word-for-word identical question after the user answered "所有" to a choice-style clarifying question. A round limit that only forced resolution after 2+ rounds still let one redundant repeat through every time.
