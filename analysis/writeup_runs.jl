#!/usr/bin/env julia
#
# Writeup analyses for the nabachd methods paper.
#
#   Part 1  per-allele neighbourhood score — BoLA alleles against the human
#           (HLA) panel, per influenza segment. nabachd's first method.
#   Part 2  symmetric panel alignment (weighted CKA + permutation null) for all
#           three pairings: BoLA–HLA, BoLA–SLA, SLA–HLA. The swine–human pair is
#           the baseline that does not involve cattle, so it says how much of
#           BoLA–HLA is generic MHC-I structure.
#   Part 3  overlap demonstration — UK vs US stem-cell-registry panels
#           (SARS-CoV-2), which share 406 alleles. Shows the metric behaves when
#           panels overlap. drop_shared stays FALSE: overlap between two registry
#           panels of the same species is real population structure, and dropping
#           it would leave a biased residual (the rarer population-specific tail).
#
#   Part 4  pooled across segments, within-segment centred — one alignment per
#           pairing over all eight segments at once, plus a pooled per-allele
#           run with a single k. See the note above part4() for why the
#           centring is done per segment block.
#
# Usage (from the repo root):
#   julia --project=. analysis/writeup_runs.jl 1
#   julia --project=. analysis/writeup_runs.jl 2
#   julia --project=. analysis/writeup_runs.jl 3
#   julia --project=. analysis/writeup_runs.jl 4
#   julia --project=. analysis/writeup_runs.jl all
#
# Results are written to RESULTS below, one CSV per part.

include(joinpath(@__DIR__, "..", "src", "nabachd.jl"))
using .nabachd
using DataFrames, CSV, Random, Statistics
using LinearAlgebra: BLAS

# ── paths ────────────────────────────────────────────────────────────────────
# Overridable from the environment so the driver can be exercised on a staged
# subset of the data (e.g. in a container) without editing this file.
const CLUST   = get(ENV, "NABACHD_CLUST",   expanduser("~/Documents/PhD/Cows/Clustering"))        # influenza H5N1
const HLAN    = get(ENV, "NABACHD_HLAN",    expanduser("~/Documents/PhD/HLA-neighbourhoods"))     # SARS-CoV-2
const RESULTS = get(ENV, "NABACHD_RESULTS", expanduser("~/Documents/PhD/nabachd_writeup/results"))

const SEGMENTS = haskey(ENV, "NABACHD_SEGMENTS") ?
    String.(split(ENV["NABACHD_SEGMENTS"], ",")) :
    ["HA", "MP", "NA", "NP", "NS", "PA", "PB1", "PB2"]

# Frequency files (see the handover for provenance).
const FREQ_HLA  = joinpath(CLUST, "supertype_panel.csv")               # 2080 alleles
const FREQ_SLA  = joinpath(CLUST, "sla_supertype_panel.csv")           #   35 alleles
# Rebuilt 2026-10-05: 18 alleles, one row each, direct carrier counting from the
# 72 Canadian Holstein bulls. Adds BoLA-3:01701 (A11 gene 3), which the previous
# file omitted, and replaces frequencies that had been read off Figure 1 of
# Codner et al. 2012 (Animal Genetics). See Cows/Clustering/cattle_panel_rebuild_notes.md.
# The HWE variant, 1-(1-p)^2 on the same counts, is supertype_panel_codner2006_hwe.csv;
# it differs by <=0.024 and reorders nothing.
const FREQ_BOLA = joinpath(CLUST, "supertype_panel_codner2006_carrier.csv")   # 18 alleles

# Permutation counts. Parts 1–2 are cheap (a few thousand variants). Part 3 is
# NOT: the US panel is 2078 alleles × ~58.7k variants, and each permutation is a
# dense product of that size — budget on the order of 10 s per permutation, so
# nperm=1000 is hours. Start at 200 to get the number, raise it for the paper.
# Overridable from the environment so a driver script can raise the count for a
# final figure without editing this file:
#   NABACHD_NPERM_PANEL=2000 NABACHD_NPERM_BIG=1000 julia --project=. analysis/writeup_runs.jl all
const NPERM_PANEL = parse(Int, get(ENV, "NABACHD_NPERM_PANEL", "1000"))
const NPERM_BIG   = parse(Int, get(ENV, "NABACHD_NPERM_BIG",   "200"))

const SEED = 42

seg_path(seg, suffix="") = joinpath(CLUST, seg * suffix, "per_allele_best_ranks.csv")

function require(path)
    isfile(path) || error("missing input: $path")
    path
end

# ── Part 1 ───────────────────────────────────────────────────────────────────
# Per-allele neighbourhood score, POOLED over the whole variant set.
#
# This reproduces the construction of the earlier HLA-neighbourhoods writeup: one
# correlation matrix of all 127 BoLA against all 175 HLA escape profiles across
# every simulated variant, a single k* chosen once from that matrix, and a
# frequency-weighted top-k neighbourhood r per BoLA allele. Not per segment —
# segments are an influenza accident and the method has to work on unsegmented
# sequences too.
#
# Note the asymmetry with Parts 4/5, and keep it: here the 127 BoLA alleles are
# the SUBJECTS, each receiving a human-relevance score, while HLA is the
# reference repertoire whose carrier frequencies make that score
# population-relevant. `weighted_nbhd_r` weights the reference side only, so all
# 127 catalogue alleles are scored and the 18-allele weighted Codner panel is not
# the right object here — it is for the panel-level CKA, where both sides are
# panels.
function part1()
    @info "Part 1 — per-allele neighbourhood score, pooled: BoLA query vs HLA reference"
    freq = load_frequencies(FREQ_HLA; on_duplicate=:error)
    q = load_cd8scape([require(seg_path(s))           for s in SEGMENTS])   # BoLA
    r = load_cd8scape([require(seg_path(s, "_human")) for s in SEGMENTS])   # HLA
    @info "  BoLA: $(nvariants(q)) variants × $(nalleles(q)) alleles; " *
          "HLA: $(nvariants(r)) variants × $(nalleles(r)) alleles"

    # panel=false: the panel metric is Part 5's job, and skipping it keeps this fast
    n = analyse(q, r; frequencies=freq, k=:auto, panel=false)
    out = copy(n.scores)
    out[!, :k_used] .= n.k

    mkpath(RESULTS)
    f = joinpath(RESULTS, "part1_per_allele_bola_vs_hla.csv")
    CSV.write(f, out)
    @info "  $(nrow(out)) BoLA alleles, k*=$(n.k)"
    @info "  → $f"

    # The k* calibration curve, so the choice of k can be shown rather than
    # asserted. k* is where the top-k score stops tracking peak r better than it
    # tracks mean r — the crossover of the two correlations.
    Mq, Mr = align_profiles(q, r)
    cal = calibrate_k(correlation_matrix(Mq, Mr))
    cal[!, :k_star] .= n.k
    f2 = joinpath(RESULTS, "part1_k_calibration.csv")
    CSV.write(f2, cal)
    @info "  → $f2  ($(nrow(cal)) k values)"

    out
end

# ── Part 2 ───────────────────────────────────────────────────────────────────
# Each pairing needs a frequency Dict spanning BOTH panels (panel_alignment
# weights each side from the same dict), so the per-species dicts are merged.
# BoLA coverage is sparse (17/127), so the all-zero-weight guard will fall back
# to uniform weights for the BoLA side and warn — expected, not an error.
function part2()
    @info "Part 2 — symmetric panel alignment, three pairings"
    f_hla, f_sla, f_bola = load_frequencies.((FREQ_HLA, FREQ_SLA, FREQ_BOLA);
                                             on_duplicate=:error)
    pairings = [
        ("BoLA", "HLA", "",       "_human", merge(f_bola, f_hla)),
        ("BoLA", "SLA", "",       "_swine", merge(f_bola, f_sla)),
        ("SLA",  "HLA", "_swine", "_human", merge(f_sla,  f_hla)),
    ]
    rows = DataFrame()
    for seg in SEGMENTS, (qn, rn, qsfx, rsfx, freq) in pairings
        q = load_cd8scape(require(seg_path(seg, qsfx)))
        r = load_cd8scape(require(seg_path(seg, rsfx)))
        pa = panel_alignment(q, r; frequencies=freq, nperm=NPERM_PANEL,
                             rng=MersenneTwister(SEED), drop_shared=false)
        # Zero-inflation check: recompute on informative variants only. Δ≈0 is
        # reassuring; a large Δ means the alignment rides on which variants are
        # active at all rather than on the escape ranking among them.
        sens = alignment_sensitivity(q, r; frequencies=freq, nperm=NPERM_PANEL,
                                     rng=MersenneTwister(SEED), drop_shared=false)
        push!(rows, (segment=seg, query=qn, reference=rn,
                     alignment=pa.alignment, z=pa.z, p=pa.p,
                     null_mean=pa.null_mean, null_sd=pa.null_sd,
                     n_query=pa.n_query, n_reference=pa.n_reference,
                     n_query_eff=pa.n_query_effective,
                     n_ref_eff=pa.n_reference_effective,
                     pr_query=pa.pr_query, pr_ref=pa.pr_reference,
                     n_variants=pa.n_variants, n_shared=pa.n_shared,
                     sens_delta=sens.delta, frac_retained=sens.frac_retained);
              promote=true)
        @info "  $seg $qn→$rn: alignment=$(round(pa.alignment, digits=4)) " *
              "z=$(round(pa.z, digits=2)) p=$(pa.p) Δsens=$(round(sens.delta, digits=4))"
    end
    mkpath(RESULTS)
    f = joinpath(RESULTS, "part2_panel_alignment.csv")
    CSV.write(f, rows); @info "  → $f  ($(nrow(rows)) rows)"
    rows
end

# ── Part 3 ───────────────────────────────────────────────────────────────────
# Part 3's panel-level statistic is the same construction as Part 5 (double
# centring, weight-matched ceiling, bootstrap CI) rather than the single-centred
# permutation version, so its value is on the same scale as the species
# comparisons and can be quoted beside them. The permutation null is not run:
# with 58,458 variants it returns its floor, exactly as in Part 2.
#
# Reps default lower than Part 5's because the matrices are an order of
# magnitude larger: cost goes as n_variants x n_alleles^2, and the US panel is
# 2,078 alleles over 58,458 variants. 200 bootstrap resamples and 100 ceiling
# draws put the ceiling's SEM near 0.006 — around a point on a "% of ceiling"
# figure, well inside the bootstrap interval.
const P3_NBOOT = parse(Int, get(ENV, "NABACHD_P3_NBOOT", "200"))
const P3_CEIL  = parse(Int, get(ENV, "NABACHD_P3_CEIL",  "100"))

function part3(; nperm = NPERM_BIG)
    @info "Part 3 — overlap demonstration: UK vs US registry panels (SARS-CoV-2)"
    uk = load_cd8scape(require(joinpath(HLAN, "UK", "per_allele_best_ranks_simulated.csv")))
    us = load_cd8scape(require(joinpath(HLAN, "US", "per_allele_best_ranks_simulated.csv")))
    # ONE DICTIONARY PER PANEL, never merged. These two panels share 406 allele
    # names, and `merge` lets the second argument win on a duplicate key, so a
    # merged dictionary weights the UK panel's shared half by US carrier
    # frequencies: HLA-A01:01 entered at 0.122 instead of 0.667, 308 of the 406
    # were down-weighted (median factor 0.11), and pr_query read 14.88 where the
    # UK panel's own frequencies give 11.73. Each panel is weighted by the
    # registry it came from.
    #
    # The species pairings in parts 2/4/5 still merge, which is safe there:
    # BoLA, HLA and SLA share no allele names, so those merges have no key
    # collisions. Part 3 is the only overlapping pair.
    freq_uk = load_frequencies(joinpath(HLAN, "UK", "supertype_panel_squished.csv");
                               on_duplicate=:error)
    freq_us = load_frequencies(joinpath(HLAN, "US", "supertype_panel_validated.csv");
                               on_duplicate=:error)

    # Per-allele scores two ways. leave_one_out=false is the primary figure;
    # =true is the diagnostic that answers "is this just alleles matching
    # themselves?" at the per-allele level.
    # `weighted_nbhd_r` weights the REFERENCE side only, so the reference
    # registry's own file is the right dictionary here. Membership is unchanged:
    # every US panel allele carrying weight under the old merged dictionary drew
    # that weight from the US file (n_ref_eff was 2078, the US file's row count).
    n_in  = analyse(uk, us; frequencies=freq_us, k=:auto, panel=false, leave_one_out=false)
    n_out = analyse(uk, us; frequencies=freq_us, k=:auto, panel=false, leave_one_out=true)

    scores = copy(n_in.scores)
    rename!(scores, :weighted_nbhd_r => :weighted_nbhd_r_with_self)
    scores[!, :weighted_nbhd_r_loo] =
        n_out.scores[indexin(scores.allele, n_out.scores.allele), :weighted_nbhd_r]

    # Panel level, twice: once over the panels as they are, once with every
    # allele the two panels share removed from both sides. The second is the
    # control the per-allele leave-one-out cannot give — leave-one-out excludes
    # an allele from its own neighbourhood, which says nothing about whether the
    # panel-level alignment is carried by the shared block.
    Mq, Mr = align_profiles(uk, us)
    nmz(x) = nabachd.normalise_allele(x)
    uk_names, us_names = nmz.(uk.alleles), nmz.(us.alleles)
    shared = intersect(Set(uk_names), Set(us_names))
    wq_all, wr_all = frequency_vector(uk, freq_uk), frequency_vector(us, freq_us)

    rows = DataFrame()
    for (tag, drop) in (("all", false), ("shared_dropped", true))
        kq = findall(i -> wq_all[i] > 0 && !(drop && uk_names[i] in shared),
                     1:length(wq_all))
        kr = findall(i -> wr_all[i] > 0 && !(drop && us_names[i] in shared),
                     1:length(wr_all))
        (isempty(kq) || isempty(kr)) &&
            error("part3: $tag left one side with no weighted allele")
        wq, wr = wq_all[kq], wr_all[kr]
        rng() = MersenneTwister(SEED)

        Sq, Sr = _prepare(Mq, kq, wq, :single), _prepare(Mr, kr, wr, :single)
        Dq, Dr = _prepare(Mq, kq, wq, :double), _prepare(Mr, kr, wr, :double)
        nq, nr = size(Dq, 2), size(Dr, 2)

        a_sw = cka(Sq, Sr; wq=wq, wr=wr)
        a_dw = cka(Dq, Dr; wq=wq, wr=wr)
        ci   = _boot_ci(Dq, Dr, wq, wr; nboot=P3_NBOOT, rng=rng())
        c_m  = _ceiling(Dr, wr, nq, wq;   rng=rng(), reps=P3_CEIL)
        c_o  = _ceiling(Dr, wr, nq, :own; rng=rng(), reps=P3_CEIL)

        ns = count(i -> uk_names[i] in shared, kq)
        push!(rows, (variant=tag, n_variants=size(Mq, 1),
                     n_query_eff=nq, n_ref_eff=nr,
                     n_shared=ns,
                     frac_shared_query=ns / nq,
                     frac_shared_reference=count(i -> us_names[i] in shared, kr) / nr,
                     pr_query=_pr(wq), pr_ref=_pr(wr),
                     align_single=a_sw,
                     align_double=a_dw, lo=ci.lo, hi=ci.hi,
                     ceiling=c_m.mean, ceiling_sd=c_m.sd,
                     ceiling_reps=P3_CEIL, ceiling_sem=c_m.sd / sqrt(P3_CEIL),
                     ceil_refweights=c_o.mean,
                     ceil_own_pr=c_o.pr_mean, ceil_own_pr_sd=c_o.pr_sd,
                     frac_ceiling=a_dw / c_m.mean,
                     frac_ceiling_lo=ci.lo / c_m.mean,
                     frac_ceiling_hi=ci.hi / c_m.mean,
                     k=n_in.k); promote=true)

        @info "  $tag: UK($nq)->US($nr) over $(size(Mq,1)) variants, $ns shared: " *
              "single=$(round(a_sw,digits=4)) double=$(round(a_dw,digits=4)) " *
              "[$(round(ci.lo,digits=4)), $(round(ci.hi,digits=4))] " *
              "ceil_own_PR=$(round(c_o.pr_mean,digits=2))±$(round(c_o.pr_sd,digits=2)) " *
              "ceiling=$(round(c_m.mean,digits=4))±$(round(c_m.sd,digits=3)) " *
              "→ $(round(100*a_dw/c_m.mean,digits=1))% of ceiling"
    end

    mkpath(RESULTS)
    f1 = joinpath(RESULTS, "part3_uk_vs_us_per_allele.csv"); CSV.write(f1, scores)
    f2 = joinpath(RESULTS, "part3_uk_vs_us_alignment.csv");  CSV.write(f2, rows)
    @info "  → $f1  ($(nrow(scores)) alleles)"
    @info "  → $f2  ($(nrow(rows)) rows)"
    (alignment=rows, scores=scores)
end

# ── Part 4 ───────────────────────────────────────────────────────────────────
# Pooled-across-segments analysis.
#
# Pooling is native: load_cd8scape accepts a list of paths and combines them,
# and the CD8scape `Frame` column holds the segment name, so variant keys stay
# globally unique across the eight segments (no collisions).
#
# The one thing pooling changes is what centring means. column_center removes
# each allele's mean across the variant axis; on a pooled matrix that is a
# single grand mean, so segment-level offsets survive — every allele in both
# panels sits higher on MP than on NA, and CKA banks that agreement before it
# looks at any within-segment ordering. That component is largely segment
# peptide composition, not BoLA.
#
# So each allele's profile is centred WITHIN each segment block before the
# matrices go downstream. Each block then has column mean 0, hence so does the
# pooled column (the block sums are each zero whatever the block sizes), which
# makes the library's own column_center a no-op and the implicit centring in
# correlation_matrix's Pearson r likewise. Nothing in src/ changes.
#
# Centring is applied AFTER variant alignment, so the blocks are exactly
# mean-zero on the matrices actually used. The returned ProfileSets share one
# key table, so the inner join inside analyse / panel_alignment is a row-for-row
# pass-through and the centring is preserved.
#
# alignment_sensitivity is deliberately NOT run here: it thresholds on zero
# entries to find zero-inflation, and centring moves the zeros. frac_retained in
# Part 2 was 1.0 (or 0.9993), so there is no zero-inflation in this dataset to
# check for anyway.

"""
    pooled_centred(q, r; frame_col="Frame") -> (ProfileSet, ProfileSet, Vector{String})

Align `q` and `r` on their shared variants, then mean-centre every allele's
profile within each `frame_col` block. Both returned sets carry the same
aligned key table; the third return value lists the segments found.
"""
function pooled_centred(q::ProfileSet, r::ProfileSet; frame_col::AbstractString="Frame")
    kc = names(q.keys)
    names(r.keys) == kc ||
        error("pooled_centred: key columns differ ($kc vs $(names(r.keys)))")
    frame_col in kc || error("pooled_centred: no '$frame_col' column in keys ($kc)")

    qk = copy(q.keys); qk.__qrow = 1:nrow(qk)
    rk = copy(r.keys); rk.__rrow = 1:nrow(rk)
    j = innerjoin(qk, rk, on=kc)
    nrow(j) == 0 && error("pooled_centred: query and reference share no variants")

    Mq = q.M[j.__qrow, :]
    Mr = r.M[j.__rrow, :]
    keys_aligned = j[:, kc]

    groups = Dict{String,Vector{Int}}()
    for (i, f) in enumerate(string.(keys_aligned[!, frame_col]))
        push!(get!(groups, f, Int[]), i)
    end
    for M in (Mq, Mr), rows in values(groups)
        blk = view(M, rows, :)
        blk .-= mean(blk, dims=1)
    end

    (ProfileSet(keys_aligned, q.alleles, Mq),
     ProfileSet(keys_aligned, r.alleles, Mr),
     sort!(collect(keys(groups))))
end

function part4()
    @info "Part 4 — pooled across segments, within-segment centred"
    f_hla, f_sla, f_bola = load_frequencies.((FREQ_HLA, FREQ_SLA, FREQ_BOLA);
                                             on_duplicate=:error)

    # Pool the eight per-segment files into one panel per species.
    @info "  loading pooled panels"
    pools = Dict(
        "BoLA" => load_cd8scape([require(seg_path(s))           for s in SEGMENTS]),
        "HLA"  => load_cd8scape([require(seg_path(s, "_human")) for s in SEGMENTS]),
        "SLA"  => load_cd8scape([require(seg_path(s, "_swine")) for s in SEGMENTS]),
    )
    for (nm, ps) in pools
        @info "    $nm: $(nvariants(ps)) variants × $(nalleles(ps)) alleles"
    end

    pairings = [
        ("BoLA", "HLA", merge(f_bola, f_hla)),
        ("BoLA", "SLA", merge(f_bola, f_sla)),
        ("SLA",  "HLA", merge(f_sla,  f_hla)),
    ]

    # ── pooled panel alignment ────────────────────────────────────────────────
    rows = DataFrame()
    for (qn, rn, freq) in pairings
        q0, r0 = pools[qn], pools[rn]
        qc, rc, frames = pooled_centred(q0, r0)
        pa = panel_alignment(qc, rc; frequencies=freq, nperm=NPERM_PANEL,
                             rng=MersenneTwister(SEED), drop_shared=false)
        push!(rows, (query=qn, reference=rn, centring="within_segment",
                     alignment=pa.alignment, z=pa.z, p=pa.p,
                     null_mean=pa.null_mean, null_sd=pa.null_sd,
                     n_query=pa.n_query, n_reference=pa.n_reference,
                     n_query_eff=pa.n_query_effective,
                     n_ref_eff=pa.n_reference_effective,
                     pr_query=pa.pr_query, pr_ref=pa.pr_reference,
                     n_variants=pa.n_variants, n_segments=length(frames),
                     n_shared=pa.n_shared); promote=true)
        @info "  $qn→$rn pooled: alignment=$(round(pa.alignment, digits=4)) " *
              "null_mean=$(round(pa.null_mean, digits=5)) p=$(pa.p) " *
              "n_variants=$(pa.n_variants)"
    end
    mkpath(RESULTS)
    f = joinpath(RESULTS, "part4_pooled_panel_alignment.csv")
    CSV.write(f, rows); @info "  → $f  ($(nrow(rows)) rows)"

    # ── pooled per-allele scores (single k over all segments) ─────────────────
    qc, rc, _ = pooled_centred(pools["BoLA"], pools["HLA"])
    n = analyse(qc, rc; frequencies=f_hla, k=:auto, panel=false)
    scores = copy(n.scores)
    scores[!, :k_used] .= n.k
    f2 = joinpath(RESULTS, "part4_pooled_per_allele_bola_vs_hla.csv")
    CSV.write(f2, scores)
    @info "  pooled per-allele: $(nrow(scores)) BoLA alleles, k=$(n.k)"
    @info "  → $f2"

    (alignment=rows, scores=scores)
end

# ── Part 5 ───────────────────────────────────────────────────────────────────
# The reconsidered panel-level test. Supersedes Parts 2 and 4: it answers their
# question on the whole variant set at once, with a statistic and a scale that
# mean something.
#
# NOT PER SEGMENT. Segments are an influenza accident — the method has to work on
# any viral sequence, and Part 3's SARS-CoV-2 panels are unsegmented already. So
# the unit is the whole variant set: the eight per-segment files are pooled into
# one panel per species (load_cd8scape takes a list of paths) and the comparison
# is made once per pairing. Nothing about `Frame` is needed, and the
# between-segment ordering that Part 2 invited — which tracked n_variants at
# r ~= -0.7 and was never interpretable — simply does not arise.
#
# Part 4's within-segment centring is also unnecessary here. A segment offset is
# a pattern in the variant means, and double centring removes the variant means,
# so the segment structure goes with them. (`part4()` is kept as the check that
# the two routes agree.)
#
# What Part 2 got wrong, visible in its own output:
#
#  (a) p is uninformative. All 24 pairings returned exactly 1/(nperm+1): no
#      permutation ever approached the observed value. Free row permutation
#      destroys all variant-level correspondence, so it tests "is there any
#      correspondence at all", which nobody doubts. Raising nperm only lowers
#      the floor.
#  (b) z is actively misleading. corr(z, n_variants) = 0.95 against
#      corr(z, alignment) = -0.17 — z measures how many variants there are,
#      because the null's sd shrinks with n. It inverted the real ordering.
#  (c) The statistic is inflated by a component both panels share for
#      uninteresting reasons. column_center removes each ALLELE's mean but
#      leaves each VARIANT's mean, and the panels agree on the variant main
#      effect ("which variants are escape-prone at all") at r ~= 0.67-0.85 —
#      the shared peptide chemistry the cows writeup already concedes.
#
# So: double centring for the statistic, a ceiling for the scale, bootstrap CIs
# instead of p.
#
# THE CEILING must be matched on WEIGHT PROFILE, not just allele count. BoLA's 18
# carrier frequencies are fairly even (participation ratio 13.1 of 18) while a
# random 18-allele HLA subset is dominated by one or two common alleles
# (PR 5.4 +/- 1.6). A count-matched ceiling carrying the reference's own weights
# is depressed by that concentration, and the observed value spuriously exceeded
# it. The ceiling therefore draws both sides from the reference repertoire at the
# two panels' own sizes AND weight profiles. `ceil_refweights` keeps the naive
# version to show what it does.
#
# It is a SCALE, NOT A BOUND: the observed compares a panel against the full
# reference while the ceiling splits that reference, so figures can exceed 100%.
#
# Panels are subset to their nonzero-weight alleles. That changes no number — a
# zero-weight column is all-zero after sqrt(w) scaling and contributes nothing to
# any inner product, verified bit-identical — but it makes the sizes honest: the
# pairings are BoLA 18 x HLA 175, BoLA 18 x SLA 35 and SLA 35 x HLA 175, not the
# 127/175/71 PanelAlignment prints.

"""
    _prepare(M_full, keep, w, mode)

Restrict to the weighted panel (`keep`), remove each allele's baseline across
variants (what `column_center` does), and for `mode === :double` also remove the
variant main effect **as this panel sees it**: the frequency-weighted mean across
the panel's own alleles, Σwᵢ·aᵢ / Σwᵢ.

The panel is the object of analysis, so the nuisance term is estimated from the
panel, weighted the same way the statistic is. Estimating it instead from all
catalogue alleles unweighted treats alleles the panel does not contain as
evidence about it; the two estimates correlate at r ≈ 0.94 and differ by ~3%,
but only the panel-local one is the quantity this metric is about.

Column means stay zero under both modes, so the `column_center` inside `cka` is
a no-op and the matrix passes through unchanged.
"""
function _prepare(M_full::AbstractMatrix, keep, w, mode::Symbol)
    A = M_full[:, keep]
    A = A .- mean(A, dims=1)
    if mode === :double
        A = A .- (A * w) ./ sum(w)          # weighted variant main effect
    end
    A
end

"The variant main effect a weighted panel sees: Σwᵢ·aᵢ / Σwᵢ per variant."
function _vme(M_full::AbstractMatrix, keep, w)
    A = M_full[:, keep]
    A = A .- mean(A, dims=1)
    vec((A * w) ./ sum(w))
end

"Columns with a positive weight — the alleles that actually contribute."
function _effective(ps::ProfileSet, freq)
    w = frequency_vector(ps, freq)
    keep = findall(>(0), w)
    if isempty(keep)
        @warn "part5: no allele of this panel carries a weight; using all alleles unweighted"
        return collect(1:nalleles(ps)), ones(nalleles(ps))
    end
    keep, w[keep]
end

"Participation ratio (Σw)²/Σw² — the effective number of alleles under weights."
_pr(w) = (p = filter(>(0), w); isempty(p) ? 0.0 : sum(p)^2 / sum(abs2, p))

"Bootstrap percentile CI, resampling shared variants."
function _boot_ci(Xq, Xr, wq, wr; nboot::Int, rng, α = 0.05)
    n = size(Xq, 1)
    vals = Vector{Float64}(undef, nboot)
    for b in 1:nboot
        idx = rand(rng, 1:n, n)
        vals[b] = cka(Xq[idx, :], Xr[idx, :]; wq = wq, wr = wr)
    end
    (lo = quantile(vals, α/2), hi = quantile(vals, 1 - α/2), sd = std(vals))
end

"""
    _ceiling(Xr, wr, nA, mode; rng, reps)

Align two disjoint random subsets of one prepared panel, the first of size `nA`.
`mode` sets how the drawn side is weighted:

- `:uniform` — both sides unweighted. No weighting confound; the primary ceiling.
- `:own`     — both sides carry the reference's own weights for the alleles
               drawn. This is the CONFOUNDED version: a small random subset of a
               registry panel is dominated by one or two common alleles, so the
               ceiling is depressed. Kept only to show that.
- a vector   — transplanted onto the drawn side, so its evenness matches the
               query panel's. The weighted analogue of `:uniform`.

Returns mean and sd over `reps` draws.
"""
function _ceiling(Xr::AbstractMatrix, wr, nA::Int, mode; rng, reps::Int)
    p = size(Xr, 2)
    # Not a clamp. Silently trimming nA would return a ceiling for a split the
    # caller did not ask for — and with :own or :uniform weights nothing further
    # would catch it, so the figure would be scaled by the wrong number. The
    # construction only makes sense when the query panel is the smaller one.
    1 <= nA < p || error("_ceiling: need 1 <= nA < $p (the reference panel's " *
                         "size); got nA = $nA. The ceiling splits the reference " *
                         "into a query-sized part and a remainder, so the query " *
                         "panel must be smaller than the reference.")

    # The fast path assumes Xr's columns are centred. Both `_prepare` modes leave
    # them that way — assert it rather than trust it, and fall back if not.
    scale   = maximum(abs, Xr)
    centred = scale == 0 || maximum(abs, mean(Xr, dims = 1)) <= 1e-8 * scale
    C2      = centred ? (Xr' * Xr) .^ 2 : nothing
    centred || @warn "_ceiling: Xr's columns are not centred; using the slow path"

    vals = Vector{Float64}(undef, reps)
    # Participation ratio of the query-sized side on each draw. Under :own this
    # is the quantity the weight-profile argument rests on — how concentrated a
    # random subset of the reference panel is — so it is reported rather than
    # recomputed by hand. Under a transplanted vector it is constant and equals
    # the query panel's own PR.
    prA  = Vector{Float64}(undef, reps)
    for i in 1:reps
        perm = randperm(rng, p)
        a, b = perm[1:nA], perm[nA+1:end]
        ua, ub = if mode === :uniform
            ones(nA), ones(length(b))
        elseif mode === :own
            wr[a], wr[b]
        else                                   # transplant the query's profile
            length(mode) == nA || error("_ceiling: transplanted weight vector is " *
                                        "$(length(mode)) long, need $nA")
            collect(mode), wr[b]
        end
        prA[i]  = _pr(ua)
        vals[i] = centred ? _cka_centred_gram(C2, a, b, ua, ub) :
                            cka(Xr[:, a], Xr[:, b]; wq = ua, wr = ub)
    end
    (mean = mean(vals), sd = std(vals), nA = nA, nB = p - nA,
     pr_mean = mean(prA), pr_sd = reps > 1 ? std(prA) : 0.0)
end

# CKA between two column subsets of one already-centred matrix, read off
# C2 = (Xr'Xr).^2. Mirrors `_weight_columns`: weights are clamped at zero, and an
# all-zero weight vector falls back to uniform for that side.
function _cka_centred_gram(C2::AbstractMatrix, a, b, ua, ub)
    u = clamp.(Float64.(ua), 0.0, Inf); all(iszero, u) && (u = ones(length(ua)))
    v = clamp.(Float64.(ub), 0.0, Inf); all(iszero, v) && (v = ones(length(ub)))
    dA  = u' * (@view(C2[a, a]) * u)
    dB  = v' * (@view(C2[b, b]) * v)
    num = v' * (@view(C2[b, a]) * u)
    den = sqrt(dA) * sqrt(dB)
    den == 0 ? 0.0 : num / den
end

"Largest alignment reached by `nperm` free row permutations — the floor check."
function _perm_max(Xq, Xr, wq, wr; nperm::Int, rng)
    maximum(cka(Xq, Xr[randperm(rng, size(Xr,1)), :]; wq=wq, wr=wr) for _ in 1:nperm)
end

const NBOOT      = parse(Int, get(ENV, "NABACHD_NBOOT",      "500"))
# The ceiling is a scale every reported figure divides by, so its own sampling
# error propagates into all of them. Its sd across draws is large (0.05-0.09),
# so the default is deliberately high: at 1000 draws the SEM is ~0.002, worth
# ~0.3 points on a "% of ceiling" figure, against ~1.5 points at 40. Lower it
# only for a quick look.
const CEIL_REPS  = parse(Int, get(ENV, "NABACHD_CEIL_REPS",  "1000"))
const FLOOR_PERM = parse(Int, get(ENV, "NABACHD_FLOOR_PERM", "200"))

function part5()
    @info "Part 5 — pooled across the whole variant set, double-centred, ceiling-scaled"
    f_hla, f_sla, f_bola = load_frequencies.((FREQ_HLA, FREQ_SLA, FREQ_BOLA);
                                             on_duplicate=:error)

    # One panel per species over every variant, not one per segment.
    @info "  pooling $(length(SEGMENTS)) input file(s) per species"
    pools = Dict(
        "BoLA" => load_cd8scape([require(seg_path(s))           for s in SEGMENTS]),
        "HLA"  => load_cd8scape([require(seg_path(s, "_human")) for s in SEGMENTS]),
        "SLA"  => load_cd8scape([require(seg_path(s, "_swine")) for s in SEGMENTS]),
    )
    for nm in ("BoLA", "HLA", "SLA")
        @info "    $nm: $(nvariants(pools[nm])) variants × $(nalleles(pools[nm])) catalogue alleles"
    end

    pairings = [("BoLA", "HLA", merge(f_bola, f_hla)),
                ("BoLA", "SLA", merge(f_bola, f_sla)),
                ("SLA",  "HLA", merge(f_sla,  f_hla))]

    rows = DataFrame()
    vme_rows = DataFrame()
    for (qn, rn, freq) in pairings
        q, r = pools[qn], pools[rn]
        kq, wq = _effective(q, freq)
        kr, wr = _effective(r, freq)
        Mq, Mr = align_profiles(q, r)
        rng() = MersenneTwister(SEED)

        vq, vr = _vme(Mq, kq, wq), _vme(Mr, kr, wr)
        vme = cor(vq, vr)
        Sq, Sr = _prepare(Mq, kq, wq, :single), _prepare(Mr, kr, wr, :single)
        Dq, Dr = _prepare(Mq, kq, wq, :double), _prepare(Mr, kr, wr, :double)
        nq, nr = size(Dq, 2), size(Dr, 2)

        a_sw = cka(Sq, Sr; wq=wq, wr=wr)
        a_dw = cka(Dq, Dr; wq=wq, wr=wr)
        ci   = _boot_ci(Dq, Dr, wq, wr; nboot=NBOOT, rng=rng())
        c_m  = _ceiling(Dr, wr, nq, wq;   rng=rng(), reps=CEIL_REPS)
        c_own= _ceiling(Dr, wr, nq, :own; rng=rng(), reps=CEIL_REPS)
        fl   = _perm_max(Dq, Dr, wq, wr; nperm=FLOOR_PERM, rng=rng())

        push!(rows, (query=qn, reference=rn,
                     n_variants=size(Mq,1), n_query_eff=nq, n_ref_eff=nr,
                     pr_query=_pr(wq), pr_ref=_pr(wr),
                     variant_main_effect_r=vme,
                     align_single=a_sw,
                     align_double=a_dw, lo=ci.lo, hi=ci.hi,
                     ceiling=c_m.mean, ceiling_sd=c_m.sd,
                     ceiling_reps=CEIL_REPS, ceiling_sem=c_m.sd/sqrt(CEIL_REPS),
                     ceil_refweights=c_own.mean,
                     ceil_own_pr=c_own.pr_mean, ceil_own_pr_sd=c_own.pr_sd,
                     frac_ceiling=a_dw/c_m.mean,
                     frac_ceiling_lo=ci.lo/c_m.mean,
                     frac_ceiling_hi=ci.hi/c_m.mean,
                     perm_max=fl, exceeds_all_perms=a_dw > fl); promote=true)

        # The agreement both panels show on "which variants are escape-prone at
        # all" — the shared peptide chemistry double centring removes. Thinned
        # for plotting; the correlation above is computed on all of them.
        stride = max(1, size(Mq,1) ÷ 4000)
        append!(vme_rows, DataFrame(query=qn, reference=rn,
                                    vme_query=vq[1:stride:end],
                                    vme_reference=vr[1:stride:end]))

        @info "  $qn($nq)→$rn($nr) over $(size(Mq,1)) variants: " *
              "single=$(round(a_sw,digits=4)) double=$(round(a_dw,digits=4)) " *
              "[$(round(ci.lo,digits=4)), $(round(ci.hi,digits=4))] " *
              "ceil_own_PR=$(round(c_own.pr_mean,digits=2))±$(round(c_own.pr_sd,digits=2)) " *
              "ceiling=$(round(c_m.mean,digits=4))±$(round(c_m.sd,digits=3)) " *
              "→ $(round(100*a_dw/c_m.mean,digits=1))% of ceiling | vme_r=$(round(vme,digits=3))"
    end
    # Panel composition as actually used: the alleles of each pooled catalogue
    # that carry a nonzero frequency. The frequency FILES are much larger than
    # this — supertype_panel.csv lists ~2,080 HLA alleles where 175 are
    # profiled — so anything describing "the panels" must read this, not the
    # files.
    comp = DataFrame()
    for (nm, fq) in (("BoLA", f_bola), ("HLA", f_hla), ("SLA", f_sla))
        ps = pools[nm]
        k, w = _effective(ps, fq)
        append!(comp, DataFrame(species = nm, allele = ps.alleles[k], weight = w,
                                n_catalogue = nalleles(ps)); promote = true)
        @info "    $nm panel: $(length(k)) of $(nalleles(ps)) catalogue alleles " *
              "carry weight (PR $(round(_pr(w), digits=1)))"
    end

    mkpath(RESULTS)
    f = joinpath(RESULTS, "part5_pooled_alignment.csv")
    CSV.write(f, rows); @info "  → $f  ($(nrow(rows)) rows)"
    f2 = joinpath(RESULTS, "part5_variant_main_effects.csv")
    CSV.write(f2, vme_rows); @info "  → $f2  ($(nrow(vme_rows)) rows, thinned)"
    f3 = joinpath(RESULTS, "part5_panel_composition.csv")
    CSV.write(f3, comp); @info "  → $f3  ($(nrow(comp)) rows)"
    rows
end

# ── entry point ──────────────────────────────────────────────────────────────
function main()
    which = isempty(ARGS) ? "all" : ARGS[1]
    which in ("1", "2", "3", "4", "5", "all") ||
        error("usage: writeup_runs.jl [1|2|3|4|5|all]")
    which in ("1", "all") && part1()
    which in ("2", "all") && part2()
    which in ("3", "all") && part3()
    which in ("4", "all") && part4()
    which in ("5", "all") && part5()
    @info "Done. Results in $RESULTS"
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
