#!/usr/bin/env bash
# One-shot paper run: 10,000 questions per method -> union -> entity/graph DBs -> mutations -> evaluation -> report + examples.
# Every step resumes if interrupted: just run the script again.
#
#   bash run_paper.sh                 # medical RAG, asks for confirmation after the dry-run
#   bash run_paper.sh --yes           # no confirmation (unattended)
#   bash run_paper.sh --pilot         # small trial in its own folders (profile `pilot`), then stops (100 per method)
#   bash run_paper.sh --pilot --n 10  # smaller pilot: 10 questions per method
#   bash run_paper.sh --dry-run       # only the plan and cost
#   bash run_paper.sh --rags medrag,lightrag,raptor     # other RAGs (need their indexes / general real queries)
#   bash run_paper.sh --heldout data/real_queries/medical_heldout.jsonl   # adds the held-out realism comparison
#   bash run_paper.sh --n 500         # 500 questions per method; mutation and evaluation then use all 500
#   MUTATE_N=3000 EVAL_N=1000 bash run_paper.sh         # sizes when --n is not given (10,000 per method from config.yaml; these are the defaults)
set -euo pipefail
export PYTHONUTF8=1   # Windows: read and write text files as UTF-8
cd "$(dirname "$0")"

RAGS=medrag; YES=0; PILOT=0; DRY=0; HELDOUT=""; N=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rags) RAGS="$2"; shift 2;;
    --n) N="$2"; shift 2;;
    --yes) YES=1; shift;;
    --pilot) PILOT=1; shift;;
    --dry-run) DRY=1; shift;;
    --heldout) HELDOUT="$2"; shift 2;;
    -h|--help) sed -n 2,12p "$0"; exit 0;;
    *) echo "Unknown option $1"; exit 2;;
  esac
done

# --n N: N questions per method; mutate and evaluate all of them. Without it: the size in config.yaml (paper profile).
if [ -n "$N" ]; then
  GEN_N="--n $N"; MUTATE_N="${MUTATE_N:-$N}"; EVAL_N="${EVAL_N:-$N}"
else
  GEN_N=""; MUTATE_N="${MUTATE_N:-3000}"; EVAL_N="${EVAL_N:-1000}"
fi

mkdir -p logs
LOG="logs/run_paper_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "==================== $(date +%H:%M:%S)  $* ===================="; }
FIRST_RAG="${RAGS%%,*}"
REAL_DOMAIN=general; [ "$RAGS" = "medrag" ] && REAL_DOMAIN=medical

if [ "$PILOT" = 1 ]; then
  PN="${N:-100}"; PL=$(( PN < 50 ? PN : 50 ))      # pilot size: --n N (default 100); mutate/evaluate at most 50 per file
  step "PILOT (profile pilot: own folders, $PN questions per method, $PL per file)"
  python scripts/run_all.py --profile pilot --rags "$RAGS" --n "$PN" --limit "$PL" \
      --steps real,generate,union,db,mutate,evaluate,analyze,audit,report,examples
  step "PILOT: are the mutated questions closer to the real ones? (feature tables, no LLM calls)"
  BP_PROFILE=pilot python scripts/compare_mutations.py
  BP_PROFILE=pilot python scripts/deviation_report.py --profile pilot
  echo "Pilot done. Read results/paper_pilot/paper_report.md, results/eval_pilot/_report/summary_counts.md and results/analysis_pilot/mutation_realism.md, then run without --pilot."
  exit 0
fi

export BP_PROFILE=paper
echo "Log: $LOG    Profile: paper    RAGs: $RAGS    n=${N:-config}  mutate-n=$MUTATE_N  eval-n=$EVAL_N"

step "0. corpora, indexes, real queries"
for r in ${RAGS//,/ }; do
  python scripts/prepare_corpora.py "$r"
  python scripts/build_indexes.py "$r"
done
python scripts/prepare_real_queries.py "$REAL_DOMAIN"

step "1. plan and cost (dry-run)"
python scripts/run_all.py --profile paper --rags "$RAGS" $GEN_N --mutate-n "$MUTATE_N" --eval-n "$EVAL_N" --dry-run
[ "$DRY" = 1 ] && exit 0
if [ "$YES" = 0 ]; then
  read -r -p "Check that every method can reach 10,000 questions and the cost is OK. Continue? [y/N] " a
  [ "$a" = y ] || [ "$a" = Y ] || { echo "Stopped."; exit 0; }
fi

step "2. generate ${N:-the configured number of} questions per method (RAGAS, ARES, DataMorgana)"
python scripts/run_all.py --profile paper --rags "$RAGS" --steps generate $GEN_N

step "3. union, entity + graph databases, corpus stats"
python scripts/run_all.py --profile paper --rags "$RAGS" --steps union,db,stats

step "4. mutations: Elaborate (m1), Colloquial (m2), Premise (m3)"
python scripts/run_all.py --profile paper --rags "$RAGS" --steps mutate --mutate-n "$MUTATE_N"

step "5. evaluation: baseline, m1, m2, m3, controls"
python scripts/run_all.py --profile paper --rags "$RAGS" --steps evaluate --eval-n "$EVAL_N"

step "6. analysis and audit"
python scripts/run_all.py --profile paper --rags "$RAGS" --steps analyze,audit

step "7. hallucination tables and plots"
python scripts/summary_counts.py
python scripts/transition_report.py
python scripts/plot_transitions.py

if [ -n "$HELDOUT" ]; then
  step "8. realism against held-out real queries"
  python scripts/compare_mutations.py --real-file "$HELDOUT" --real-domain "$REAL_DOMAIN"
  python scripts/deviation_report.py --profile paper --real-file "$HELDOUT"
fi

step "9. paper report and example queries"
python scripts/run_all.py --profile paper --rags "$RAGS" --steps report,examples

step "DONE"
echo "Report:   results/paper_final/paper_report.md  (+ paper_tables.tex, story_${FIRST_RAG}.csv)"
echo "Examples: results/paper_final/paper_examples.md / .tex, example_graphs_${FIRST_RAG}.pdf"
echo "Audit:    results/paper_final/audit_sheet_mutations.csv"
echo "Log:      $LOG"
