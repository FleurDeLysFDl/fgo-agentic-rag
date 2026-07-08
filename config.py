from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
WIKI_RAW_DIR = DATA_DIR / "wiki_raw"
QUEST_RAW_DIR = DATA_DIR / "quest_raw"
QUEST_SCRIPTS_DIR = QUEST_RAW_DIR / "scripts_txt"
# JP-exclusive story content (quests not yet ported to CN -- new events, main
# story chapters, and existing servants' new interludes) kept in a separate
# manifest/script dir since the raw text here is untranslated Japanese, unlike
# everything else in the corpus.
QUEST_SCRIPTS_DIR_JP = QUEST_RAW_DIR / "scripts_txt_jp"
CORPUS_DIR = DATA_DIR / "corpus"
SERVANTS_DB_PATH = DATA_DIR / "servants.db"
CONVERSATIONS_DB_PATH = DATA_DIR / "conversations.db"
QDRANT_PATH = DATA_DIR / "qdrant_local"
QDRANT_COLLECTION = "servants_corpus"

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.viaaai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

for d in (RAW_DIR, WIKI_RAW_DIR, CORPUS_DIR, QUEST_RAW_DIR, QUEST_SCRIPTS_DIR, QUEST_SCRIPTS_DIR_JP):
    d.mkdir(parents=True, exist_ok=True)

# wars that are pure farming/grinding stages with no story content at all
# (verified: their quests all have empty phaseScripts) -- excluded from the
# quest-script corpus. Everything else (main story, Lostbelts, Ordeal Call,
# interludes/幕间物语, and all limited-time event/collab story content) is
# in scope per explicit user decision.
QUEST_EXCLUDE_WAR_IDS = {1001, 1002, 1006, 9999}

ATLAS_API_BASE = "https://api.atlasacademy.io"
ATLAS_STRUCTURED_REGION = os.environ.get("ATLAS_STRUCTURED_REGION", "JP")
ATLAS_NAME_REGION = os.environ.get("ATLAS_NAME_REGION", "CN")

WIKI_API_BASE = os.environ.get("WIKI_API_BASE", "https://fgo.wiki/api.php")
WIKI_REQUEST_DELAY_SECONDS = float(os.environ.get("WIKI_REQUEST_DELAY_SECONDS", "1.0"))
WIKI_USER_AGENT = os.environ.get(
    "WIKI_USER_AGENT",
    "fgo-agentic-rag-research-bot/0.1 (personal portfolio project)",
)
