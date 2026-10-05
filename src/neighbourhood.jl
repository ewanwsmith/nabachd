# neighbourhood.jl — the MHC-I neighbourhood metric.
#
# Given a query panel of MHC-I alleles and a (larger, optionally
# frequency-weighted) reference panel, each described by CD8scape escape
# profiles across a shared set of variants, we quantify how well each query
# allele's escape landscape is mirrored by the reference repertoire.
#
# Pipeline:
#   1. R = Pearson correlation of every query profile against every reference
#      profile  ->  (n_query × n_reference) matrix.
#   2. Per query allele: peak_r  = max over reference,
#                        mean_r  = mean over reference.
#   3. Neighbourhood size k* selected by the crossover criterion
#      (see `find_k_star`), or fixed by the user.
#   4. Frequency-weighted top-k neighbourhood r (arcsin averaged) — the
#      primary metric.

# Pre-averaging transform.
#
# arcsin (2·asin, mapped to [-π, π]) is bounded everywhere INCLUDING at r = ±1,
# where Fisher's z diverges. That matters whenever query and reference panels
# share alleles: an allele scored in both panels on identical inputs correlates
# at exactly 1.0, and Fisher's z could only handle it by clamping (atanh of
# 0.999999 ≈ 7.25), giving such alleles ~5× the leverage of a genuinely strong
# r = 0.9. arcsin needs no clamp: 2·asin(1) = π.
#
# The "arcsin is outdated" critique (Warton & Hui 2011) concerns arcsin(√p) for
# proportions, which is a different transform on a different quantity.
arcsin_t(r)    = 2 * asin(clamp(r, -1.0, 1.0))
arcsin_tinv(z) = sin(z / 2)

const TRANSFORMS = (:arcsin, :none)

"""
    correlation_matrix(Mq, Mr) -> R

Pearson correlation between every column of `Mq` (query profiles) and every
column of `Mr` (reference profiles). Returns an `n_query × n_reference` matrix.
`Mq` and `Mr` must be aligned on the same variants (same number of rows).
"""
function correlation_matrix(Mq::AbstractMatrix, Mr::AbstractMatrix)
    size(Mq, 1) == size(Mr, 1) ||
        error("correlation_matrix: query and reference have different variant counts")
    R = cor(Mq, Mr)
    # A zero-variance (constant / never-binding on the shared variants) profile has
    # undefined Pearson r (NaN); treat it as 0 (no linear similarity) so a single
    # degenerate allele can't poison the whole matrix.
    @inbounds for i in eachindex(R)
        isfinite(R[i]) || (R[i] = 0.0)
    end
    return R
end

"""
    topk_mean(r, k)

Unweighted mean of the `k` largest correlations in `r`. This is the top-k
neighbourhood score used for k-calibration.
"""
function topk_mean(r::AbstractVector, k::Int)
    k = clamp(k, 1, length(r))
    mean(partialsort(r, 1:k, rev=true))
end

"""
    weighted_neighbourhood_r(r, freq, k; transform=:arcsin)

Frequency-weighted top-k neighbourhood r for one query allele.

`r` is the vector of correlations of that allele against all reference alleles;
`freq` are the reference-allele weights aligned to `r`. The `k` most-correlated
reference alleles are selected, their weights renormalised within the
neighbourhood, and (by default) the correlations are arcsin-transformed before
the weighted mean and back-transformed — reducing the leverage of near-unity
correlations. If all `k` weights are zero, uniform weights are used.

`transform` is `:arcsin` (default) or `:none` for a plain weighted mean.
"""
function weighted_neighbourhood_r(r::AbstractVector, freq::AbstractVector, k::Int; transform::Symbol=:arcsin)
    transform in TRANSFORMS ||
        error("weighted_neighbourhood_r: transform must be :arcsin or :none (got :$transform)")
    k = clamp(k, 1, length(r))
    idx = partialsortperm(r, 1:k, rev=true)
    w = Float64.(freq[idx])
    s = sum(w)
    w = s > 0 ? w ./ s : fill(1.0 / k, k)
    rk = r[idx]
    transform === :arcsin ? arcsin_tinv(sum(w .* arcsin_t.(rk))) : sum(w .* rk)
end

"""
    default_k_sequence(n_ref) -> Vector{Int}

The sweep of neighbourhood sizes used for k-calibration: every integer up to
min(50, n_ref) for a dense view of the crossover region, plus a log-spaced tail
out to `n_ref`.
"""
function default_k_sequence(n_ref::Int)
    dense = collect(1:min(50, n_ref))
    tail  = floor.(Int, exp.(range(log(2), log(n_ref), length=60)))
    sort(unique(vcat(dense, tail, n_ref)))
end

"""
    calibrate_k(R; k_seq=default_k_sequence(size(R,2))) -> DataFrame

Sweep neighbourhood size `k`. For each `k`, compute the top-k neighbourhood
score for every query allele and correlate it (across query alleles) with the
per-allele peak r and mean r, and record its spread. Columns:
`k, cor_peak, cor_mean, sd_topk`.
"""
function calibrate_k(R::AbstractMatrix; k_seq=default_k_sequence(size(R, 2)))
    n_q = size(R, 1)
    peak = vec(maximum(R, dims=2))
    meanr = vec(mean(R, dims=2))
    cor_peak = Float64[]; cor_mean = Float64[]; sd_topk = Float64[]
    for k in k_seq
        tk = [topk_mean(view(R, i, :), k) for i in 1:n_q]
        push!(cor_peak, cor(tk, peak))
        push!(cor_mean, cor(tk, meanr))
        push!(sd_topk, std(tk))
    end
    DataFrame(k=collect(k_seq), cor_peak=cor_peak, cor_mean=cor_mean, sd_topk=sd_topk)
end

"""
    find_k_star(R; k_seq=...) -> (k_star::Int, calibration::DataFrame)

Choose the neighbourhood size k* as the crossover where the top-k score is
equally predictive of peak r and mean r:

    k* = argzero_k [ cor(topk, peak_r) − cor(topk, mean_r) ]

The crossover is located by the first sign change of the difference over the
sweep and linearly interpolated between the bracketing k values. k* is the
largest integer at which the top-k score is still more correlated with peak r
than with mean r — i.e. `floor` of the interpolated crossover, the last
peak-dominant neighbourhood size. If no crossover exists, the k minimising
|Δcor| is returned (with a warning).
"""
function find_k_star(R::AbstractMatrix; k_seq=default_k_sequence(size(R, 2)))
    cal = calibrate_k(R; k_seq=k_seq)
    d = cal.cor_peak .- cal.cor_mean
    ci = findfirst(i -> sign(d[i]) != sign(d[i+1]), 1:length(d)-1)
    if ci === nothing
        k_star = cal.k[argmin(abs.(d))]
        @warn "find_k_star: no crossover found; using argmin|Δcor| → k = $k_star"
    else
        klo, khi = cal.k[ci], cal.k[ci+1]
        dlo, dhi = d[ci], d[ci+1]
        k_cross = klo + (-dlo) / (dhi - dlo) * (khi - klo)
        k_star = floor(Int, k_cross)
    end
    return k_star, cal
end

"""
    neighbourhood_scores(R, query_alleles, reference_alleles, freq; k,
                         transform=:arcsin, leave_one_out=false) -> DataFrame

Compute per-query-allele summary statistics and neighbourhood scores. Returns a
DataFrame sorted by descending weighted neighbourhood r, with columns:
`allele, k, peak_r, mean_r, nbhd_r, weighted_nbhd_r, nearest_reference, nearest_r,
shared, nearest_is_self`.

- `nbhd_r`          — unweighted top-k mean correlation
- `weighted_nbhd_r` — frequency-weighted, arcsin top-k mean (primary metric)
- `nearest_reference`/`nearest_r` — closest single reference allele and its r
  (always over the full reference, not affected by `leave_one_out`)
- `shared`          — `true` when the query allele's normalised name appears in
                      the reference panel (diagnostic, always computed)
- `nearest_is_self` — `true` when the nearest reference allele is a name-match
                      for this query allele (flags potential self-inflation)
- `leave_one_out`   — when `true`, exclude reference alleles whose normalised
                      name matches the query allele before computing `nbhd_r` and
                      `weighted_nbhd_r`; `k` in the output reflects the effective
                      neighbourhood size used (may be < requested `k` only when
                      fewer than `k` non-matching reference alleles exist)
"""
function neighbourhood_scores(R::AbstractMatrix, query_alleles::AbstractVector,
                              reference_alleles::AbstractVector, freq::AbstractVector;
                              k::Int, transform::Symbol=:arcsin, leave_one_out::Bool=false)
    n_q, n_ref = size(R)
    n_q == length(query_alleles) ||
        error("neighbourhood_scores: R rows ($n_q) ≠ query alleles ($(length(query_alleles)))")
    n_ref == length(reference_alleles) == length(freq) ||
        error("neighbourhood_scores: R cols ($n_ref) must match reference alleles and freq")

    norm_ref     = normalise_allele.(reference_alleles)
    norm_ref_set = Set(norm_ref)

    out = DataFrame(allele=String[], k=Int[], peak_r=Float64[], mean_r=Float64[],
                    nbhd_r=Float64[], weighted_nbhd_r=Float64[],
                    nearest_reference=String[], nearest_r=Float64[],
                    shared=Bool[], nearest_is_self=Bool[])
    for i in 1:n_q
        r     = R[i, :]
        jbest = argmax(r)
        qnorm = normalise_allele(query_alleles[i])

        is_shared       = qnorm ∈ norm_ref_set
        nearest_is_self = norm_ref[jbest] == qnorm

        if leave_one_out && is_shared
            keep  = findall(j -> norm_ref[j] != qnorm, 1:n_ref)
            r_loo = r[keep]
            f_loo = freq[keep]
            k_eff = min(k, length(keep))
        else
            r_loo = r
            f_loo = freq
            k_eff = k
        end

        push!(out, (string(query_alleles[i]), k_eff,
                    maximum(r), mean(r),
                    topk_mean(r_loo, k_eff),
                    weighted_neighbourhood_r(r_loo, f_loo, k_eff; transform=transform),
                    string(reference_alleles[jbest]), r[jbest],
                    is_shared, nearest_is_self))
    end
    sort!(out, :weighted_nbhd_r, rev=true)
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# Panel-level neighbourhood metric: frequency-weighted linear CKA.
#
# `weighted_nbhd_r` is a per-allele metric — for each query allele, how well its
# escape profile is mirrored by the reference repertoire. The panel-level
# question is different: do the two panels organise the *variant* escape
# landscape the same way? Centred kernel alignment (CKA) answers exactly that.
# It is symmetric, bounded in [0, 1], needs no allele correspondence between
# panels (it is invariant to rotation / relabelling of the allele basis) and is
# defined for panels of different sizes — so CKA(BoLA, HLA) and CKA(BoLA, SLA)
# sit on the same scale. We call this panel statistic the *neighbourhood
# alignment*.
#
# Linear CKA (Kornblith et al. 2019) between two column-centred profile matrices
# X (m variants × n_q alleles) and Y (m × n_r):
#
#     CKA(X, Y) = ‖Yᵀ X‖_F²  /  (‖Xᵀ X‖_F · ‖Yᵀ Y‖_F)
#
# Frequency weighting scales each allele column by √wₐ before the products, so
# XᵀX becomes the carrier-frequency-weighted Gram Σₐ wₐ xₐ xₐᵀ and common alleles
# dominate the geometry. A global rescaling of the weights cancels; only the
# relative carrier frequencies matter.

"""
    column_center(M) -> Matrix

Mean-centre each column of `M` — centre every allele's escape profile across the
shared variants. The centring step of linear CKA.
"""
column_center(M::AbstractMatrix) = M .- mean(M, dims=1)

# Scale columns by √w (so XᵀX is the w-weighted Gram). Weights are clamped to be
# non-negative; a zero weight simply drops that allele from the geometry. If a
# panel's weights are *all* zero (e.g. a frequency file that covers none of its
# alleles), fall back to uniform (unweighted) for that panel rather than zeroing
# every column — which would otherwise collapse the alignment to 0.
function _weight_columns(X::AbstractMatrix, w)
    w === nothing && return X
    length(w) == size(X, 2) ||
        error("cka: weight length $(length(w)) ≠ number of alleles $(size(X, 2))")
    ww = clamp.(Float64.(w), 0.0, Inf)
    if all(iszero, ww)
        @warn "cka: all weights are zero for a panel (frequency file covers none of its alleles); using uniform weights for that panel"
        return X
    end
    X .* sqrt.(ww)'
end

# CKA from already centred (+ weighted) matrices. Kept separate so the
# permutation null centres / weights once and only reshuffles the cross term.
function _cka(Xc::AbstractMatrix, Yc::AbstractMatrix)
    den = sqrt(sum(abs2, Xc'Xc)) * sqrt(sum(abs2, Yc'Yc))
    den == 0 ? 0.0 : sum(abs2, Yc'Xc) / den
end

"""
    cka(Mq, Mr; wq=nothing, wr=nothing) -> Float64

Linear centred kernel alignment between two escape-profile matrices sharing the
same `m` variants (rows). `Mq` is `m × n_q`, `Mr` is `m × n_r`; the allele counts
need not match. Each allele profile is mean-centred across variants, and optional
per-allele weights `wq` / `wr` (e.g. carrier frequencies) scale the columns by
√w. Returns a value in [0, 1] — 1 when the panels induce the same
variant-similarity structure (up to rotation and isotropic scaling of the allele
basis), 0 when unrelated. Symmetric: `cka(Mq, Mr) == cka(Mr, Mq)`.
"""
function cka(Mq::AbstractMatrix, Mr::AbstractMatrix; wq=nothing, wr=nothing)
    size(Mq, 1) == size(Mr, 1) ||
        error("cka: query and reference have different variant counts")
    _cka(_weight_columns(column_center(Mq), wq),
         _weight_columns(column_center(Mr), wr))
end

# ─────────────────────────────────────────────────────────────────────────────
# Shared-allele diagnostics.
#
# When two panels overlap (alleles with the same normalised name appear in both),
# there are two effects:
#   • per-allele: the nearest reference neighbour for a shared query allele may
#     be itself, inflating weighted_nbhd_r. `leave_one_out=true` in
#     `neighbourhood_scores` removes these matches before computing the score.
#   • panel: shared alleles contribute identically to both Gram matrices,
#     mechanically increasing CKA regardless of the panels' biological
#     similarity. `drop_shared=true` in `panel_alignment` removes them first.
#
# Both flags are **off by default** — shared alleles are a legitimate signal
# (they force a minimum alignment reflecting the shared portion of the repertoire,
# which is itself meaningful). The flags are for sensitivity checks and for
# comparisons where shared alleles would make the result trivially high.
#
# Shared-allele statistics are **always reported** (they cost nothing to compute)
# so the user can judge whether to apply the flags.

# Internal: compute shared-allele overlap statistics from allele-name vectors and
# (optionally) weight vectors. n_query / n_reference are the ORIGINAL panel sizes
# (used as denominators); if weights are provided they should be aligned to the
# ORIGINAL allele vectors (before any dropping).
function _shared_info(query_alleles, reference_alleles, wq, wr, n_query, n_reference)
    (query_alleles === nothing || reference_alleles === nothing) &&
        return (n_shared=0, frac_shared_query=0.0, frac_shared_reference=0.0,
                frac_shared_query_weighted=0.0, frac_shared_reference_weighted=0.0)
    qnorm = normalise_allele.(query_alleles)
    rnorm = normalise_allele.(reference_alleles)
    shared_set = intersect(Set(qnorm), Set(rnorm))
    n_sh = length(shared_set)
    fq   = n_sh / max(1, n_query)
    fr   = n_sh / max(1, n_reference)
    # frequency-weighted fractions: shared allele weight / total weight
    if wq !== nothing
        wq_total = sum(wq)
        fqw = wq_total > 0 ?
            sum(w for (w, nm) in zip(wq, qnorm) if nm ∈ shared_set; init=0.0) / wq_total : fq
    else
        fqw = fq
    end
    if wr !== nothing
        wr_total = sum(wr)
        frw = wr_total > 0 ?
            sum(w for (w, nm) in zip(wr, rnorm) if nm ∈ shared_set; init=0.0) / wr_total : fr
    else
        frw = fr
    end
    return (n_shared=n_sh, frac_shared_query=fq, frac_shared_reference=fr,
            frac_shared_query_weighted=fqw, frac_shared_reference_weighted=frw)
end

"""
    PanelAlignment

Panel-level neighbourhood alignment — a single frequency-weighted CKA between a
query and reference panel with its permutation-null significance.

**Primary metric:**
- `alignment`  — frequency-weighted linear CKA in [0, 1]
- `p`          — one-sided empirical p, `P(null ≥ observed)`
- `z`          — standardised effect `(observed − null mean) / null sd`;
                 comparable across panel pairs of different sizes
- `null_mean`, `null_sd`, `nperm` — the permutation-null summary

**Panel metadata:**
- `weighted`       — whether carrier frequencies were applied
- `n_query`, `n_reference`, `n_variants` — the aligned panel dimensions
- `drop_shared`    — whether shared alleles were excluded before computing

**Shared-allele diagnostics** (always populated; 0 / 0.0 when names unavailable):
- `n_shared`                    — alleles with the same normalised name in both panels
- `frac_shared_query`           — n_shared / n_query (unweighted)
- `frac_shared_reference`       — n_shared / n_reference (unweighted)
- `frac_shared_query_weighted`  — shared allele frequency / total query frequency
- `frac_shared_reference_weighted` — same for the reference panel
"""
struct PanelAlignment
    alignment::Float64
    p::Float64
    z::Float64
    null_mean::Float64
    null_sd::Float64
    nperm::Int
    weighted::Bool
    n_query::Int
    n_reference::Int
    n_variants::Int
    drop_shared::Bool
    n_shared::Int
    frac_shared_query::Float64
    frac_shared_reference::Float64
    frac_shared_query_weighted::Float64
    frac_shared_reference_weighted::Float64
end

Base.show(io::IO, pa::PanelAlignment) = print(io,
    "PanelAlignment(alignment=$(round(pa.alignment, digits=4)), " *
    "z=$(round(pa.z, digits=2)), p=$(pa.p), " *
    "$(pa.weighted ? "frequency-weighted" : "unweighted"), " *
    "$(pa.n_query)×$(pa.n_reference) alleles, $(pa.n_variants) variants" *
    (pa.n_shared > 0 ?
        ", $(pa.n_shared) shared ($(round(100*pa.frac_shared_query, digits=1))% Q / $(round(100*pa.frac_shared_reference, digits=1))% R)" :
        "") *
    (pa.drop_shared ? " [shared alleles excluded]" : "") * ")")

"""
    panel_alignment(Mq, Mr; wq=nothing, wr=nothing, nperm=1000,
                    rng=Random.default_rng(), weighted=false,
                    n_query=size(Mq,2), n_reference=size(Mr,2),
                    query_alleles=nothing, reference_alleles=nothing,
                    drop_shared=false) -> PanelAlignment

Neighbourhood alignment between two aligned escape-profile matrices, with a
permutation null. The null shuffles the reference's variant rows relative to the
query — breaking the variant alignment while preserving each panel's own
structure. A row permutation leaves each panel's own Gram matrix unchanged, so
the denominator is computed once and only the cross term is recomputed per
permutation. Returns a [`PanelAlignment`](@ref).

**Shared-allele handling:**
- `query_alleles` / `reference_alleles` — allele name vectors aligned to the
  columns of `Mq` / `Mr`. When provided, shared-allele statistics are reported
  in the result even when `drop_shared=false`.
- `drop_shared=true` — remove alleles whose normalised name appears in both
  panels *before* computing the alignment. Requires `query_alleles` and
  `reference_alleles`. Shared-allele statistics always reflect the *original*
  panel sizes (before dropping), so the effect of the flag is legible from the
  difference in the scores.
"""
function panel_alignment(Mq::AbstractMatrix, Mr::AbstractMatrix;
                         wq=nothing, wr=nothing, nperm::Int=1000,
                         rng=Random.default_rng(), weighted::Bool=false,
                         n_query::Int=size(Mq, 2), n_reference::Int=size(Mr, 2),
                         query_alleles=nothing, reference_alleles=nothing,
                         drop_shared::Bool=false)
    size(Mq, 1) == size(Mr, 1) ||
        error("panel_alignment: query and reference have different variant counts")

    # ── shared-allele diagnostics (always computed from original allele lists) ──
    si = _shared_info(query_alleles, reference_alleles, wq, wr, n_query, n_reference)

    # ── optional: drop shared alleles from both panels before CKA ──────────────
    Mq_use, Mr_use, wq_use, wr_use = Mq, Mr, wq, wr
    if drop_shared && query_alleles !== nothing && reference_alleles !== nothing && si.n_shared > 0
        qnorm = normalise_allele.(query_alleles)
        rnorm = normalise_allele.(reference_alleles)
        shared_set = intersect(Set(qnorm), Set(rnorm))
        keep_q = findall(nm -> nm ∉ shared_set, qnorm)
        keep_r = findall(nm -> nm ∉ shared_set, rnorm)
        isempty(keep_q) && error("panel_alignment: drop_shared removed all query alleles")
        isempty(keep_r) && error("panel_alignment: drop_shared removed all reference alleles")
        Mq_use = Mq[:, keep_q]
        Mr_use = Mr[:, keep_r]
        wq_use = wq !== nothing ? wq[keep_q] : nothing
        wr_use = wr !== nothing ? wr[keep_r] : nothing
    end

    # ── CKA + permutation null ──────────────────────────────────────────────────
    Xc = _weight_columns(column_center(Mq_use), wq_use)
    Yc = _weight_columns(column_center(Mr_use), wr_use)
    obs = _cka(Xc, Yc)
    m   = size(Xc, 1)
    den = sqrt(sum(abs2, Xc'Xc)) * sqrt(sum(abs2, Yc'Yc))   # invariant to row perm
    null = Vector{Float64}(undef, nperm)
    for b in 1:nperm
        Yp = Yc[randperm(rng, m), :]
        null[b] = den == 0 ? 0.0 : sum(abs2, Yp'Xc) / den
    end
    p = (count(>=(obs), null) + 1) / (nperm + 1)
    μ = mean(null); σ = std(null)
    z = σ > 0 ? (obs - μ) / σ : 0.0

    PanelAlignment(obs, p, z, μ, σ, nperm, weighted, n_query, n_reference, m,
                   drop_shared, si.n_shared, si.frac_shared_query,
                   si.frac_shared_reference, si.frac_shared_query_weighted,
                   si.frac_shared_reference_weighted)
end

"""
    panel_alignment(query::ProfileSet, reference::ProfileSet;
                    frequencies=nothing, nperm=1000, rng=Random.default_rng(),
                    drop_shared=false)

Convenience method: align `query` and `reference` on their shared variants and
compute the neighbourhood alignment. With `frequencies` (a `Dict` from
`load_frequencies`, keyed across both panels) both sides are weighted by carrier
frequency; without it the alignment is unweighted. Set `drop_shared=true` to
exclude alleles shared between the two panels before computing.
"""
function panel_alignment(query::ProfileSet, reference::ProfileSet;
                         frequencies=nothing, nperm::Int=1000,
                         rng=Random.default_rng(), drop_shared::Bool=false)
    Mq, Mr = align_profiles(query, reference)
    weighted = frequencies !== nothing
    wq = weighted ? frequency_vector(query, frequencies)     : nothing
    wr = weighted ? frequency_vector(reference, frequencies) : nothing
    panel_alignment(Mq, Mr; wq=wq, wr=wr, nperm=nperm, rng=rng, weighted=weighted,
                    n_query=nalleles(query), n_reference=nalleles(reference),
                    query_alleles=query.alleles, reference_alleles=reference.alleles,
                    drop_shared=drop_shared)
end

# ─────────────────────────────────────────────────────────────────────────────
# Zero-inflation sensitivity: escape profiles fill non-binding / structurally
# absent (variant, allele) cells with 0. A variant that is non-binding across a
# whole panel is an all-zero row; a large block of such rows shifts every
# allele's column mean at centring and lets the panels agree merely on *which*
# variants are immunologically active at all, inflating the alignment above what
# the fine-grained escape ranking among active variants would give. The check:
# recompute the alignment on the informative variants only (those carrying
# binding signal) and compare. Entry-level masking is ill-defined for a global
# matrix metric like CKA, so masking is at the variant (row) level, where the
# zero-fill actually lives.

"""
    informative_variants(Mq, Mr; threshold=0.0, mode=:either) -> BitVector

Row mask over the shared variants. A variant is *informative* when its peak
absolute escape score exceeds `threshold` in the query and/or reference panel,
per `mode`:

- `:either`    — active in at least one panel (default)
- `:both`      — active in both panels (strictest; keeps only jointly-tested,
                 binding-relevant variants)
- `:query` / `:reference` — active in that panel only

Non-informative variants are the non-binding / structural zeros; masking them is
the zero-inflation sensitivity check for [`cka`](@ref) / [`panel_alignment`](@ref).
"""
function informative_variants(Mq::AbstractMatrix, Mr::AbstractMatrix;
                              threshold::Real=0.0, mode::Symbol=:either)
    size(Mq, 1) == size(Mr, 1) ||
        error("informative_variants: query and reference have different variant counts")
    q = vec(maximum(abs, Mq, dims=2))
    r = vec(maximum(abs, Mr, dims=2))
    if     mode === :either;    (q .> threshold) .| (r .> threshold)
    elseif mode === :both;      (q .> threshold) .& (r .> threshold)
    elseif mode === :query;     q .> threshold
    elseif mode === :reference; r .> threshold
    else   error("informative_variants: mode must be :either, :both, :query or :reference")
    end
end

"""
    AlignmentSensitivity

Zero-inflation sensitivity of the panel neighbourhood alignment: the alignment
on all shared variants versus on the informative variants only.

- `full`, `masked` — the two [`PanelAlignment`](@ref)s (all variants / masked)
- `mode`, `threshold` — the masking rule used
- `n_variants_full`, `n_variants_masked`, `frac_retained` — how many rows survived
- `delta` — `masked.alignment − full.alignment`; near 0 means the shared zeros
  were not driving the score, a large negative delta means they were
"""
struct AlignmentSensitivity
    full::PanelAlignment
    masked::PanelAlignment
    mode::Symbol
    threshold::Float64
    n_variants_full::Int
    n_variants_masked::Int
    frac_retained::Float64
    delta::Float64
end

Base.show(io::IO, s::AlignmentSensitivity) = print(io,
    "AlignmentSensitivity(full=$(round(s.full.alignment, digits=4)), " *
    "masked=$(round(s.masked.alignment, digits=4)), " *
    "Δ=$(round(s.delta, digits=4)), " *
    "$(s.n_variants_masked)/$(s.n_variants_full) variants kept " *
    "($(round(100*s.frac_retained, digits=1))%), mode=:$(s.mode))")

"""
    alignment_sensitivity(Mq, Mr; wq=nothing, wr=nothing, threshold=0.0,
                          mode=:either, nperm=1000, rng=Random.default_rng(),
                          weighted=false, n_query=size(Mq,2),
                          n_reference=size(Mr,2),
                          query_alleles=nothing, reference_alleles=nothing,
                          drop_shared=false) -> AlignmentSensitivity

Compute the neighbourhood alignment on all shared variants and again on the
informative variants only (see [`informative_variants`](@ref)), returning both
with the fraction of variants retained and their difference. The masked run uses
the same weights, permutation count and RNG so the two are directly comparable.
`query_alleles`, `reference_alleles`, and `drop_shared` are forwarded to
`panel_alignment` (shared-allele diagnostics appear in both `full` and `masked`).
"""
function alignment_sensitivity(Mq::AbstractMatrix, Mr::AbstractMatrix;
                               wq=nothing, wr=nothing, threshold::Real=0.0,
                               mode::Symbol=:either, nperm::Int=1000,
                               rng=Random.default_rng(), weighted::Bool=false,
                               n_query::Int=size(Mq, 2), n_reference::Int=size(Mr, 2),
                               query_alleles=nothing, reference_alleles=nothing,
                               drop_shared::Bool=false)
    full = panel_alignment(Mq, Mr; wq=wq, wr=wr, nperm=nperm, rng=rng,
                           weighted=weighted, n_query=n_query, n_reference=n_reference,
                           query_alleles=query_alleles, reference_alleles=reference_alleles,
                           drop_shared=drop_shared)
    mask = informative_variants(Mq, Mr; threshold=threshold, mode=mode)
    keep = count(mask)
    keep >= 2 ||
        error("alignment_sensitivity: only $keep informative variant(s) at threshold=$threshold, mode=:$mode")
    masked = panel_alignment(Mq[mask, :], Mr[mask, :]; wq=wq, wr=wr, nperm=nperm,
                             rng=rng, weighted=weighted, n_query=n_query,
                             n_reference=n_reference,
                             query_alleles=query_alleles, reference_alleles=reference_alleles,
                             drop_shared=drop_shared)
    AlignmentSensitivity(full, masked, mode, Float64(threshold), size(Mq, 1), keep,
                         keep / size(Mq, 1), masked.alignment - full.alignment)
end

"""
    alignment_sensitivity(query::ProfileSet, reference::ProfileSet;
                          frequencies=nothing, threshold=0.0, mode=:either,
                          nperm=1000, rng=Random.default_rng(),
                          drop_shared=false)

Convenience method: align `query` and `reference` on their shared variants, then
run the zero-inflation sensitivity check. `frequencies` weights both panels by
carrier frequency, as in [`panel_alignment`](@ref). Set `drop_shared=true` to
exclude shared alleles before computing (forwarded to both `panel_alignment` calls).
"""
function alignment_sensitivity(query::ProfileSet, reference::ProfileSet;
                               frequencies=nothing, threshold::Real=0.0,
                               mode::Symbol=:either, nperm::Int=1000,
                               rng=Random.default_rng(), drop_shared::Bool=false)
    Mq, Mr = align_profiles(query, reference)
    weighted = frequencies !== nothing
    wq = weighted ? frequency_vector(query, frequencies)     : nothing
    wr = weighted ? frequency_vector(reference, frequencies) : nothing
    alignment_sensitivity(Mq, Mr; wq=wq, wr=wr, threshold=threshold, mode=mode,
                          nperm=nperm, rng=rng, weighted=weighted,
                          n_query=nalleles(query), n_reference=nalleles(reference),
                          query_alleles=query.alleles, reference_alleles=reference.alleles,
                          drop_shared=drop_shared)
end

# ─────────────────────────────────────────────────────────────────────────────
# Escape concordance — genotype-level neighbourhood metric.
#
# For a patient's ≤6-allele HLA genotype (a ProfileSet of profiles), escape
# concordance quantifies how well the patient's escape landscape agrees with the
# frequency-weighted local reference panel.
#
# Per-allele concordance:   weighted_nbhd_r (Fisher-z top-k, leave_one_out=true)
# Genotype concordance (E): unweighted mean of per-allele scores
# k selection:              derived from the reference self-neighbourhood, keeping
#                           patient and null scores on the same k and avoiding
#                           instability over the tiny (≤6) query
# Null calibration:         nperm random frequency-proportional local genotypes
#                           (locus-structured, heterozygous) scored by the same
#                           pipeline; z, percentile, and two-sided p reported.

# Internal: weighted single draw from a probability vector (which must sum to 1).
function _sample_weighted(rng, w::AbstractVector{<:Real})
    u = rand(rng)
    s = 0.0
    for (i, wi) in enumerate(w)
        s += wi
        s >= u && return i
    end
    return length(w)      # rounding fallback
end

# Internal: derive k* from the reference panel's own self-neighbourhood.
# Applies the same crossover criterion as find_k_star to the reference
# self-correlation matrix with the diagonal masked to -1 (LOO), treating
# each reference allele as a query against the rest of the panel.
# k* is the neighbourhood size at which topk_mean transitions from tracking
# the single-best-allele peak to tracking the panel mean — the crossover point
# of cor(topk, peak) and cor(topk, mean) as k increases.
function _k_star_reference(R_ref::AbstractMatrix;
                            k_seq=default_k_sequence(size(R_ref, 2)))
    n  = size(R_ref, 1)
    R  = copy(R_ref)
    for i in 1:n; R[i, i] = -1.0; end      # mask self-match
    peak  = vec(maximum(R, dims=2))
    meanr = vec(mean(R, dims=2))
    ks    = collect(k_seq)
    cor_peak = Float64[]; cor_mean = Float64[]
    for k in ks
        tk = [topk_mean(view(R, i, :), k) for i in 1:n]
        push!(cor_peak, cor(tk, peak))
        push!(cor_mean, cor(tk, meanr))
    end
    d  = cor_peak .- cor_mean
    ci = findfirst(i -> sign(d[i]) != sign(d[i+1]), 1:length(d)-1)
    if ci === nothing
        k_star = ks[argmin(abs.(d))]
        @warn "_k_star_reference: no crossover found; using argmin|Δcor| → k = $k_star"
    else
        klo, khi = ks[ci], ks[ci+1]
        dlo, dhi = d[ci], d[ci+1]
        k_cross  = klo + (-dlo) / (dhi - dlo) * (khi - klo)
        k_star   = floor(Int, k_cross)
    end
    return k_star
end

"""
    sample_local_genotypes(reference::ProfileSet, frequencies::AbstractDict;
                           n=1000, loci=["A","B","C"],
                           match_zygosity=true,
                           rng=Random.default_rng()) -> Vector{Vector{String}}

Draw `n` random genotypes from the local reference panel, sampling each locus
independently with probability proportional to carrier frequency. Only alleles
that both have a profile in `reference` AND have frequency > 0 in `frequencies`
are eligible; frequencies are renormalised within each locus over that eligible set.

With `match_zygosity=true` (default) each locus contributes two *distinct*
alleles (no homozygotes), matching a fully-heterozygous diploid; the first draw
is removed before the second is made. When a locus has only one eligible allele,
or when `match_zygosity=false`, two copies of the same allele are placed.

`loci` names the loci to sample. The default `["A","B","C"]` is the classical
HLA class I set; pass `:auto` to infer the loci present in `reference` instead,
which is what any non-HLA panel (BoLA, SLA) needs — their locus tokens are digits
or letters other than A/B/C, so the default matches nothing and every locus is
dropped. An explicit vector is still honoured.

Returns a `Vector{Vector{String}}` — each inner vector holds the allele names
for one sampled genotype (up to 2 × length(loci) names).
"""
function sample_local_genotypes(reference::ProfileSet, frequencies::AbstractDict;
                                 n::Int                  = 1000,
                                 loci::Union{AbstractVector,Symbol} = ["A", "B", "C"],
                                 match_zygosity::Bool    = true,
                                 rng                     = Random.default_rng())
    norm_ref = normalise_allele.(reference.alleles)

    # Parse the locus token from a normalised class I allele name: the single
    # character after the species prefix. HLA-A01:01 → "A", BoLA-2:01201 → "2",
    # BoLA-N:03401 → "N", SLA-1:0101 → "1". For HLA class I this returns exactly
    # what the previous HLA-only pattern did. Class II names (HLA-DRB1…) carry a
    # multi-character locus and are not handled.
    function _locus(name::AbstractString)
        m = match(r"^[A-Za-z]+-([A-Za-z0-9])", name)
        m === nothing ? nothing : String(m[1])
    end

    # `loci=:auto` — take the loci actually present in the reference panel.
    loci_use = if loci === :auto
        sort(unique(String[l for l in (_locus(nm) for nm in norm_ref) if l !== nothing]))
    else
        string.(loci)
    end
    isempty(loci_use) && error("sample_local_genotypes: could not determine any locus " *
                               "from the reference allele names")

    # Build per-locus eligible allele lists (must be profiled AND have freq > 0).
    locus_alleles  = Dict{String, Vector{String}}()
    locus_weights  = Dict{String, Vector{Float64}}()
    for locus in loci_use
        als = String[]; wts = Float64[]
        for (allele, norm) in zip(reference.alleles, norm_ref)
            _locus(norm) == locus || continue
            w = get(frequencies, norm, 0.0)
            w > 0 || continue
            push!(als, allele); push!(wts, w)
        end
        if isempty(als)
            @warn "sample_local_genotypes: no profiled allele with frequency > 0 for locus $locus; locus omitted from null genotypes"
            continue
        end
        locus_alleles[locus] = als
        locus_weights[locus] = wts ./ sum(wts)
    end

    # Every locus dropped means the null genotypes would all be empty, which
    # silently yields NaN concordance scores downstream. Fail loudly instead.
    isempty(locus_alleles) && error(
        "sample_local_genotypes: no profiled allele with frequency > 0 for any of " *
        "loci $(loci_use). For a non-HLA panel pass loci=:auto or the panel's own " *
        "locus names; the default [\"A\",\"B\",\"C\"] is HLA class I only.")

    genotypes = Vector{Vector{String}}(undef, n)
    for i in 1:n
        g = String[]
        for locus in loci_use
            !haskey(locus_alleles, locus) && continue
            als = locus_alleles[locus]; wts = locus_weights[locus]
            idx1 = _sample_weighted(rng, wts)
            push!(g, als[idx1])
            if match_zygosity && length(als) >= 2
                w2 = copy(wts); w2[idx1] = 0.0; s = sum(w2)
                idx2 = s > 0 ? _sample_weighted(rng, w2 ./ s) : idx1
                push!(g, als[idx2])
            else
                push!(g, als[idx1])       # allow homozygote
            end
        end
        genotypes[i] = g
    end
    return genotypes
end

"""
    EscapeConcordance

Escape concordance of a patient HLA genotype with a local reference panel.

**Per-allele table** (`per_allele::DataFrame`, sorted by concordance descending):
One row per genotype allele with columns
`allele, k, peak_r, mean_r, nbhd_r, weighted_nbhd_r, nearest_reference, nearest_r,
shared, nearest_is_self`. `weighted_nbhd_r` is the per-allele escape concordance —
frequency-weighted arcsin top-k neighbourhood r (see `weighted_neighbourhood_r`),
by default with leave-one-out applied so an allele shared with the reference panel
does not inflate its own score.

**Genotype summary:**
- `E`      — unweighted mean of per-allele `weighted_nbhd_r` (the primary score)
- `E_min`, `E_max` — per-allele extremes (surfaces an idiosyncratic allele)

**Null calibration** (from `nperm` random locus-structured frequency-proportional
genotypes scored by the same pipeline):
- `z`          — `(E − null_mean) / null_sd`
- `percentile` — fraction of null genotypes with E ≤ E_patient
- `p`          — two-sided empirical p (`2 × min(tail, 1−tail)`, +1/+1 smoothed)
- `null_mean`, `null_sd`, `nperm`

**Dimensions and settings:**
- `k`, `k_source` — neighbourhood size and derivation (`:auto` = reference-derived,
                    `:user` = manually supplied)
- `n_genotype`, `n_reference`, `n_variants` — panel and variant counts
"""
struct EscapeConcordance
    per_allele   ::DataFrame
    E            ::Float64
    E_min        ::Float64
    E_max        ::Float64
    z            ::Float64
    percentile   ::Float64
    p            ::Float64
    null_mean    ::Float64
    null_sd      ::Float64
    nperm        ::Int
    k            ::Int
    k_source     ::Symbol
    n_genotype   ::Int
    n_reference  ::Int
    n_variants   ::Int
end

Base.show(io::IO, ec::EscapeConcordance) = print(io,
    "EscapeConcordance(E=$(round(ec.E, digits=4)), " *
    "z=$(round(ec.z, digits=2)), percentile=$(round(100*ec.percentile, digits=1))%, " *
    "k=$(ec.k) [$(ec.k_source)], " *
    "$(ec.n_genotype) genotype × $(ec.n_reference) reference alleles, " *
    "$(ec.n_variants) variants)")

"""
    escape_concordance(query::ProfileSet, reference::ProfileSet;
                       frequencies=nothing, k::Union{Int,Symbol}=:auto,
                       transform=:arcsin, nperm=1000,
                       loci=["A","B","C"], match_zygosity=true,
                       leave_one_out_query=true, leave_one_out_null=true,
                       alternative=:two_sided,
                       rng=Random.default_rng()) -> EscapeConcordance

Escape concordance of the `query` genotype against the `reference` local panel,
with null calibration from random locus-structured local genotypes.

`query` is a ProfileSet of the patient's allele profiles (e.g. loaded from the
patient panel's `per_allele_best_ranks.csv` with `load_cd8scape`). `reference`
is the local population panel (e.g. the AFND Moscow supertype panel). Both are
aligned on their shared variants before scoring.

**Arguments:**
- `frequencies` — `Dict{String,Float64}` from `load_frequencies`, keyed on
  normalised allele names. Weights the per-allele neighbourhood metric and the
  null genotype sampler. With `nothing`, scoring is unweighted and the null
  samples uniformly within each locus over profiled alleles.
- `k` — neighbourhood size: `:auto` (default) derives k* from the reference
  panel's own self-neighbourhood structure, keeping patient and null scores on
  the same footing; pass an `Int` to fix it.
- `transform` — pre-averaging transform for the weighted neighbourhood r:
  `:arcsin` (default) or `:none`.
- `nperm` — number of random local genotypes for the null (default `1000`).
- `loci`, `match_zygosity` — passed to [`sample_local_genotypes`](@ref) for the
  null. The `loci` default is HLA class I; a non-HLA reference panel needs
  `loci=:auto` or an explicit list, or no null genotype can be built.
- `leave_one_out_query`, `leave_one_out_null` — whether a genotype allele that is
  also in the reference panel is excluded from its own neighbourhood, for the
  patient and the null respectively. Both default `true`, which is symmetric.
  Setting `leave_one_out_query=false` with the null left at `true` gives the
  asymmetric variant: a shared allele contributes its genuine signal to the
  patient score, while within-null self-matches (r = 1.0) are still prevented
  from inflating the null. That asymmetry is untested — it changes what `E`
  means relative to its null, so compare both before relying on it.
- `alternative` — tail for `p`: `:two_sided` (default), `:less` for the
  hypothesis that concordance is unusually *low* (the direction of interest when
  asking whether a genotype is poorly covered by the local panel), or
  `:greater`. `z` and `percentile` are unaffected.
- `rng` — e.g. `MersenneTwister(42)` for reproducibility.

Returns an [`EscapeConcordance`](@ref).
"""
function escape_concordance(query::ProfileSet, reference::ProfileSet;
                            frequencies = nothing,
                            k::Union{Int, Symbol} = :auto,
                            transform::Symbol      = :arcsin,
                            nperm::Int             = 1000,
                            loci::Union{AbstractVector,Symbol} = ["A", "B", "C"],
                            match_zygosity::Bool   = true,
                            leave_one_out_query::Bool = true,
                            leave_one_out_null::Bool  = true,
                            alternative::Symbol    = :two_sided,
                            rng                    = Random.default_rng())
    alternative in (:two_sided, :less, :greater) || error(
        "escape_concordance: alternative must be :two_sided, :less or :greater " *
        "(got $alternative)")

    # ── Align on shared variants ────────────────────────────────────────────
    Mq, Mr  = align_profiles(query, reference)
    n_vars  = size(Mq, 1)
    n_ref   = nalleles(reference)

    freq_vec = frequencies !== nothing ? frequency_vector(reference, frequencies) :
                                         ones(n_ref)

    # ── Reference self-correlation (computed once; reused for k* and null) ──
    R_self = cor(Mr, Mr)     # (n_ref × n_ref), diagonal = 1.0

    # ── k selection ─────────────────────────────────────────────────────────
    k_source = :auto
    if k === :auto
        k = _k_star_reference(R_self)
    else
        k_source = :user
    end
    k_use = Int(k)

    # ── Per-allele concordance for the patient genotype ─────────────────────
    R      = correlation_matrix(Mq, Mr)
    scores = neighbourhood_scores(R, query.alleles, reference.alleles, freq_vec;
                                  k=k_use, transform=transform,
                                  leave_one_out=leave_one_out_query)
    wscores = scores.weighted_nbhd_r
    E     = mean(wscores)
    E_min = minimum(wscores)
    E_max = maximum(wscores)

    # ── Null: random locus-structured frequency-proportional genotypes ───────
    # When no frequencies are supplied, sample uniformly within each locus over
    # all profiled alleles (assign weight 1.0 to every reference allele).
    null_freq_dict = if frequencies !== nothing
        frequencies
    else
        Dict(normalise_allele(a) => 1.0 for a in reference.alleles)
    end

    null_genotypes = sample_local_genotypes(reference, null_freq_dict;
                                             n=nperm, loci=loci,
                                             match_zygosity=match_zygosity, rng=rng)

    # Build a lookup: normalised allele name → column index in R_self
    norm_ref       = normalise_allele.(reference.alleles)
    ref_name_index = Dict(nm => i for (i, nm) in enumerate(norm_ref))

    null_E = Float64[]
    sizehint!(null_E, nperm)
    for null_g in null_genotypes
        norm_null = normalise_allele.(null_g)
        idxs = [get(ref_name_index, nm, nothing) for nm in norm_null]
        any(isnothing, idxs) && continue      # skip unprofiled (should not happen)
        idxs    = Int.(idxs)
        R_null  = R_self[idxs, :]             # (n_null_alleles × n_ref)
        s_null  = neighbourhood_scores(R_null, null_g, reference.alleles, freq_vec;
                                       k=k_use, transform=transform,
                                       leave_one_out=leave_one_out_null)
        push!(null_E, mean(s_null.weighted_nbhd_r))
    end

    # ── Null calibration ────────────────────────────────────────────────────
    n_null    = length(null_E)
    null_mean = n_null > 0 ? mean(null_E) : 0.0
    null_sd   = n_null > 1 ? std(null_E)  : 1.0
    z         = null_sd > 0 ? (E - null_mean) / null_sd : 0.0
    pctile    = n_null > 0 ? count(<=(E), null_E) / n_null : 0.5
    # Permutation p with +1/+1 smoothing (mirrors panel_alignment).
    upper     = (count(>=(E), null_E) + 1) / (n_null + 1)   # P(null >= E)
    lower     = (count(<=(E), null_E) + 1) / (n_null + 1)   # P(null <= E)
    p = alternative === :two_sided ? min(2 * min(upper, lower), 1.0) :
        alternative === :less      ? lower :
                                     upper

    EscapeConcordance(scores, E, E_min, E_max, z, pctile, p,
                      null_mean, null_sd, nperm,
                      k_use, k_source,
                      nalleles(query), n_ref, n_vars)
end
