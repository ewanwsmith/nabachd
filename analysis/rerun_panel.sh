#!/usr/bin/env bash
# =============================================================================
# rerun_panel.sh — re-run the nabachd writeup analyses after a BoLA panel change
#
# Parts 1/2/4 take minutes. Part 3 is the overnight one (see §5 of the handover).
#
# NO netMHCpan / CD8scape step is needed: every allele in the rebuilt BoLA panel,
# including the newly added BoLA-3:01701, is already present in all eight
# per_allele_best_ranks.csv files, which hold the full 127-allele netMHCpan
# catalogue. The frequency file supplies weights only.
#
#   DO NOT run run_simulate_supertype_all.sh as part of this. With an 18-allele
#   panel it would replace those 127-allele per-allele files with 18-allele ones,
#   which breaks Part 1 and changes what Parts 2/4 see.
#
# Usage, from anywhere:
#   bash ~/Documents/PhD/Software/nabachd/analysis/rerun_panel.sh
#
#   PARTS=5          run only these parts, comma-separated, in this order
#                    (default 1,2,4,5,3 — Part 3 last because it is the slow one)
#   SKIP_PART3=1     drop Part 3 from whatever PARTS asks for
#   PART3_NPERM=1000 permutations for Part 3 (default 200)
#   PANEL_NPERM=1000 permutations for Parts 2/4/5 (default 1000)
#   MAKE_FIGS=0            skip the R figure build at the end (default: build)
#   NABACHD_NBOOT=500      Part 5 bootstrap draws for the CIs
#   NABACHD_CEIL_REPS=1000 Part 5 ceiling draws (default is high on purpose —
#                          every reported figure is divided by the ceiling)
#
# Unattended:
#   nohup bash ~/Documents/PhD/Software/nabachd/analysis/rerun_panel.sh >/dev/null 2>&1 &
#   tail -f ~/Documents/PhD/nabachd_writeup/logs/rerun_*.log
# =============================================================================
[[ -z "${CAFFEINATED:-}" ]] && exec env CAFFEINATED=1 caffeinate -ims bash \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")" "$@"
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$HOME/Documents/PhD/nabachd_writeup/results"
LOGS="$HOME/Documents/PhD/nabachd_writeup/logs"

PART3_NPERM="${PART3_NPERM:-200}"
PANEL_NPERM="${PANEL_NPERM:-1000}"
SKIP_PART3="${SKIP_PART3:-0}"
PARTS="${PARTS:-1,2,4,5,3}"
MAKE_FIGS="${MAKE_FIGS:-1}"
FIG_DIR="$HOME/Documents/PhD/nabachd_writeup/figures"

mkdir -p "$LOGS" "$RESULTS"
MASTER="${LOGS}/rerun_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

step() {   # step <name> <cmd...>   (nperm comes from the exports below)
    local name="$1"; shift
    local t0=$SECONDS
    log "── ${name}"
    if "$@" >>"$MASTER" 2>&1; then
        log "   ✓ ${name}  ($(( SECONDS - t0 ))s)"
    else
        log "   ✗ ${name} FAILED after $(( SECONDS - t0 ))s — see ${MASTER}"
        exit 1
    fi
}

cd "$REPO"
log "repo    : ${REPO} @ $(git rev-parse --short HEAD) ($(git status --porcelain | wc -l | tr -d ' ') files dirty)"
# Read the const itself, not the first filename anywhere in the file — the
# comment above FREQ_BOLA names the HWE variant, which an unanchored grep picks
# up first and misreports.
log "panel   : $(sed -n 's/^const FREQ_BOLA.*"\(.*\)".*/\1/p' analysis/writeup_runs.jl | xargs basename)"
log "parts   : ${PARTS}"
log "nperm   : Parts 2/4/5 ${PANEL_NPERM}, Part 3 ${PART3_NPERM}"
log "log     : ${MASTER}"
log ""

export NABACHD_NPERM_PANEL="$PANEL_NPERM"
export NABACHD_NPERM_BIG="$PART3_NPERM"

# Gate on the suite: 278 expected (209 main + 29 escape_concordance
# + 5 sample_local_genotypes + 20 ec kwargs + 15 load_frequencies duplicates).
step "test suite" julia --project=. test/runtests.jl

# No associative arrays: macOS ships bash 3.2.
part_label() {
    case "$1" in
        1) echo "Part 1  per-allele" ;;
        2) echo "Part 2  per-segment" ;;
        3) echo "Part 3  overlap" ;;
        4) echo "Part 4  pooled" ;;
        5) echo "Part 5  reconsidered" ;;
        *) echo "Part $1" ;;
    esac
}

IFS=',' read -r -a PART_LIST <<< "$PARTS"
for pt in "${PART_LIST[@]}"; do
    pt="$(echo "$pt" | tr -d '[:space:]')"
    [[ -z "$pt" ]] && continue
    if [[ "$pt" == "3" && "$SKIP_PART3" == "1" ]]; then
        log "── Part 3 skipped (SKIP_PART3=1)"
        continue
    fi
    [[ "$pt" == "3" ]] && log "── Part 3 next — 2078 alleles x ~58.7k variants, 6.6 GB load, slow at nperm=${PART3_NPERM}"
    step "$(part_label "$pt")" julia --project=. analysis/writeup_runs.jl "$pt"
done

log ""
log "══ done in $(( SECONDS / 60 ))m — results in ${RESULTS}"
ls -lt "$RESULTS" | sed -n '2,12p' | tee -a "$MASTER"
# ── Figures ──────────────────────────────────────────────────────────────────
# Built from the result CSVs, never recomputing the metric: nabachd is the
# single source of truth for every number in the writeup.
if [[ "$MAKE_FIGS" == "1" ]]; then
    if ! command -v Rscript >/dev/null 2>&1; then
        log "── figures SKIPPED: Rscript not on PATH"
    elif [[ ! -f "${FIG_DIR}/make_figures.R" ]]; then
        log "── figures SKIPPED: ${FIG_DIR}/make_figures.R not found"
    else
        step "figures (R/ggplot)" Rscript "${FIG_DIR}/make_figures.R"
    fi
fi

log ""
log "Worth checking first:"
case ",$PARTS," in *,2,*|*,4,*) 
    log "  · pooled n_variants still equals the per-segment sum (22778 for BoLA-HLA)"
    log "  · quote alignment against null_mean, never z — z tracks n_variants (r=0.95)" ;;
esac
case ",$PARTS," in *,5,*)
    log "  · compare pairings WITHIN a segment: n_variants is held fixed there."
    log "    Between-segment ordering still tracks n (r~-0.7) and is not interpretable."
    log "  · the ceiling is a scale, not a bound — figures can exceed 100%" ;;
esac
