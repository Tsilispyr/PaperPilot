#!/usr/bin/env bash
# =============================================================================
# PaperPilot - Interactive Setup & Launch
#
# Χρήση (από WSL terminal, στον φάκελο του project):
#   bash start.sh
#
# Το script:
#   1. Ρωτά ποιον AI πάροχο θέλεις (OpenAI / Google / Ollama)
#   2. Ρωτά το προσωπικό σου API key
#   3. Ενημερώνει το .env
#   4. Τρέχει docker compose up -d
#   5. Δείχνει όλες τις διευθύνσεις & credentials
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT/.env"

#  ANSI ─
B=$'\e[1m'   # bold
R=$'\e[0m'   # reset
C=$'\e[36m'  # cyan
G=$'\e[32m'  # green
Y=$'\e[33m'  # yellow
D=$'\e[2m'   # dim
RE=$'\e[31m' # red

#  helpers ─
die()  { echo "${RE}ΣΦΑΛΜΑ: $*${R}" >&2; exit 1; }
ok()   { echo "${G}✓ $*${R}"; }
info() { echo "${D}  $*${R}"; }

sed_replace() {
    # sed_replace KEY VALUE FILE  - αντικαθιστά KEY=... στο FILE
    local key="$1" val="$2" file="$3"
    # Χρήση | ως delimiter ώστε τα / στα URLs να μην σπάνε την εντολή
    sed -i "s|^${key}=.*|${key}=${val}|" "$file"
}

# ========= Banner =========
clear
echo ""
echo "${B}${C}  ╔══════════════════════════════════════════════════╗${R}"
echo "${B}${C}  ║              P a p e r P i l o t                 ║${R}"
echo "${B}${C}  ║       Agentic RAG System - Setup & Launch        ║${R}"
echo "${B}${C}  ╚══════════════════════════════════════════════════╝${R}"
echo ""

# ========= Έλεγχος .env (δημιουργία αν δεν υπάρχει) =========
if [[ ! -f "$ENV_FILE" ]]; then
    info "Δεν βρέθηκε .env - δημιουργία με προεπιλεγμένες τιμές..."
    cat > "$ENV_FILE" << 'ENVEOF'
LLM_PROVIDER=openai

OPENAI_API_KEY=your-openai-api-key-here
OPENAI_LLM_MODEL=gpt-4.1-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_JUDGE_MODEL=gpt-4.1-mini

GEMINI_API_KEY=your-gemini-api-key-here
GOOGLE_LLM_MODEL=gemini-2.0-flash
GOOGLE_EMBED_MODEL=text-embedding-004

OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_LLM_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=bge-m3

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_V1=papers_v1
QDRANT_COLLECTION_V2=papers_v2

LANGFUSE_PUBLIC_KEY=pk-lf-paperpilot-2026-pub
LANGFUSE_SECRET_KEY=sk-lf-paperpilot-2026-sec
LANGFUSE_HOST=http://localhost:3001
LANGFUSE_PROJECT=paperpilot

NEXTAUTH_SECRET=f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8
SALT=1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b
ENCRYPTION_KEY=1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b
LANGFUSE_INIT_ORG_NAME=PaperPilot
LANGFUSE_INIT_PROJECT_NAME=paperpilot
LANGFUSE_INIT_USER_EMAIL=admin@paperpilot.local
LANGFUSE_INIT_USER_NAME=Admin
LANGFUSE_INIT_USER_PASSWORD=PaperPilot2026!

POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=postgres
CLICKHOUSE_USER=clickhouse
CLICKHOUSE_PASSWORD=clickhouse
MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=miniosecret

ARXIV_MAX_PAPERS=150
ARXIV_FROM_DATE=2020-01-01
ARXIV_TO_DATE=2026-04-30
ARXIV_SEED_IDS=2212.10496,2210.03629,2310.11511,2309.15217,2408.08067,2404.16130,2309.07597,2005.11401,2004.04906,2112.09118

CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=50

TOP_K_V1=5
TOP_K_V2_DENSE=8
TOP_K_V2_RERANK=4
RERANKER_MODEL=BAAI/bge-reranker-base

AGENT_MAX_ITERATIONS=6
CACHE_DB_PATH=data/cache.db
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
ENVEOF
    ok ".env δημιουργήθηκε"
fi

#  Βήμα 1: Επιλογή παρόχου 
echo "${B} Βήμα 1: Επιλογή AI παρόχου ─${R}"
echo ""
echo "  1)  OpenAI   │ gpt-4.1-mini · text-embedding-3-small"
echo "  2)  Google   │ gemini-2.0-flash · text-embedding-004"
echo "  3)  Ollama   │ τοπικά μοντέλα (πρέπει να τρέχει στο host)"
echo ""

while true; do
    read -rp "  Επιλογή [1/2/3]: " choice
    case "$choice" in
        1) PROVIDER="openai";  break ;;
        2) PROVIDER="google";  break ;;
        3) PROVIDER="ollama";  break ;;
        *) echo "  ${Y}Παρακαλώ πληκτρολόγησε 1, 2 ή 3.${R}" ;;
    esac
done

echo ""

# ========= Βήμα 2: API Key / URL =========
echo "${B} Βήμα 2: Credentials παρόχου ─${R}"
echo ""

case "$PROVIDER" in
    openai)
        echo "  Χρειάζεσαι: OpenAI API Key (αρχίζει με  sk-...)"
        echo "  Από: https://platform.openai.com/api-keys"
        echo ""
        while true; do
            read -rp "  OpenAI API Key: " API_KEY
            [[ -n "$API_KEY" ]] && break
            echo "  ${Y}Το key δεν μπορεί να είναι κενό.${R}"
        done
        sed_replace "LLM_PROVIDER"   "openai"   "$ENV_FILE"
        sed_replace "OPENAI_API_KEY" "$API_KEY" "$ENV_FILE"
        ok ".env → LLM_PROVIDER=openai, OPENAI_API_KEY ενημερώθηκε"
        ;;

    google)
        echo "  Χρειάζεσαι: Google AI Studio API Key (αρχίζει με  AIza...)"
        echo "  Από: https://aistudio.google.com/app/apikey"
        echo ""
        while true; do
            read -rp "  Gemini API Key: " API_KEY
            [[ -n "$API_KEY" ]] && break
            echo "  ${Y}Το key δεν μπορεί να είναι κενό.${R}"
        done
        sed_replace "LLM_PROVIDER"  "google"  "$ENV_FILE"
        sed_replace "GEMINI_API_KEY" "$API_KEY" "$ENV_FILE"
        ok ".env → LLM_PROVIDER=google, GEMINI_API_KEY ενημερώθηκε"
        ;;

    ollama)
        echo "  Χρειάζεσαι: Ollama να τρέχει στο host σου"
        echo "  Εγκατάσταση: https://ollama.ai  →  ollama serve"
        echo ""
        read -rp "  Ollama URL [http://host.docker.internal:11434]: " OLLAMA_URL
        OLLAMA_URL="${OLLAMA_URL:-http://host.docker.internal:11434}"
        sed_replace "LLM_PROVIDER"  "ollama"      "$ENV_FILE"
        sed_replace "OLLAMA_BASE_URL" "$OLLAMA_URL" "$ENV_FILE"
        ok ".env → LLM_PROVIDER=ollama, OLLAMA_BASE_URL=$OLLAMA_URL"
        ;;
esac

echo ""

# ========= Βήμα 3: docker compose up =========
echo "${B} Βήμα 3: Εκκίνηση υπηρεσιών ${R}"
echo ""

cd "$ROOT"

# ========= Έλεγχος ύπαρξης docker =========
command -v docker &>/dev/null || die "docker δεν βρέθηκε. Εγκατάστησε Docker Desktop."

docker compose up -d

# ========= Summary - εμφανίζεται ΑΜΕΣΩΣ μετά το compose =========
echo ""
echo "${B}${C}  ╔══════════════════════════════════════════════════════════╗${R}"
echo "${B}${C}  ║        Containers ξεκίνησαν - URLs & Credentials         ║${R}"
echo "${B}${C}  ╚══════════════════════════════════════════════════════════╝${R}"
echo ""
echo "  ${B}PaperPilot Chat UI${R}  ${D}(~30s για να φορτώσει)${R}"
echo "    ${G}http://localhost:8000${R}"
echo "    Πάροχος : ${C}${PROVIDER}${R}"
echo ""
echo "  ${B}Langfuse${R}  ${D}(tracing & observability)${R}"
echo "    ${G}http://localhost:3001${R}"
echo "    Χρήστης : admin@paperpilot.local   Κωδικός : PaperPilot2026!"
echo "    ${D}Project 'paperpilot' δημιουργείται αυτόματα - δεν χρειάζεται ρύθμιση${R}"
echo ""
echo "  ${B}Qdrant${R}  ${D}(vector database dashboard)${R}"
echo "    ${G}http://localhost:6333${R}"
echo ""
echo "  ${B}MinIO${R}  ${D}(blob storage console)${R}"
echo "    ${G}http://localhost:9091${R}"
echo "    Χρήστης : minio   Κωδικός : miniosecret"
echo ""
echo "${D}  Logs app   :  docker compose logs -f app${R}"
echo "${D}  Stop όλα   :  docker compose down${R}"
echo "${D}  Αλλαγή AI  :  bash start.sh${R}"
echo ""

# ========= Αναμονή app + Qdrant check =========
printf "${D}  Αναμονή app"
ATTEMPTS=0
until curl -sf http://localhost:8000 &>/dev/null; do
    ATTEMPTS=$((ATTEMPTS+1))
    [[ $ATTEMPTS -ge 25 ]] && break
    printf "${D}.${R}"
    sleep 2
done
echo "${R}"

QDRANT_RESP=$(curl -sf http://localhost:6333/collections 2>/dev/null || echo "{}")
COUNT=$(echo "$QDRANT_RESP" | grep -o '"name"' | wc -l)

if [[ "$COUNT" -eq 0 ]]; then
    echo ""
    echo "${Y}  ┌───────────────────────────────────────────────────────────┐${R}"
    echo "${Y}  │  ΠΡΩΤΗ ΕΚΚΙΝΗΣΗ - χρειάζεται ingestion των papers         │${R}"
    echo "${Y}  │                                                           │${R}"
    echo "${Y}  │  Τρέξε σε νέο terminal:                                   │${R}"
    echo "${Y}  │    docker compose exec app python -m paperpilot.ingest    │${R}"
    echo "${Y}  │                                                           │${R}"
    echo "${Y}  │  Διάρκεια: ~5-10 λεπτά (κατέβασμα + embedding 50 papers)  │${R}"
    echo "${Y}  └───────────────────────────────────────────────────────────┘${R}"
else
    ok "Έτοιμο! Qdrant: ${COUNT} collection(s) φορτωμένα - άνοιξε http://localhost:8000"
fi
echo ""
