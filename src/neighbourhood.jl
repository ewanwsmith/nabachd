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
#   4. Frequency-weighted top-k neighbourhood r (Fisher-z averaged) — the
#      primary metric.

fisher_z(r)    = atanh(clamp(r, -0.999999, 0.999999))
fisher_zinv(z) = tanh(z)

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
    weighted_neighbourhood_r(r, freq, k; fisher=true)

Frequency-weighted top-k neighbourhood r for one query allele.

`r` is the vector of correlations of that allele against all reference alleles;
`freq` are the reference-allele weights aligned to `r`. The `k` most-correlated
reference alleles are selected, their weights renormalised within the
neighbourhood, and (by default) the correlations are Fisher z-transformed before
the weighted mean and back-transformed — reducing the leverage of near-unity
correlations. If all `k` weights are zero, uniform weights are used.
"""
function weighted_neighbourhood_r(r::AbstractVector, freq::AbstractVector, k::Int; fisher::Bool=true)
    k = clamp(k, 1, length(r))
    idx = partialsortperm(r, 1:k, rev=true)
    w = Float64.(freq[idx])
    s = sum(w)
    w = s > 0 ? w ./ s : fill(1.0 / k, k)
    rk = r[idx]
    fisher ? fisher_zinv(sum(w .* fisher_z.(rk))) : sum(w .* rk)
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
    neighbourhood_scores(R, query_alleles, reference_alleles, freq; k, fisher=true) -> DataFrame

Compute per-query-allele summary statistics and neighbourhood scores. Returns a
DataFrame sorted by descending weighted neighbourhood r, with columns:
`allele, k, peak_r, mean_r, nbhd_r, weighted_nbhd_r, nearest_reference, nearest_r`.

- `nbhd_r`          — unweighted top-k mean correlation
- `weighted_nbhd_r` — frequency-weighted, Fisher-z top-k mean (primary metric)
- `nearest_reference`/`nearest_r` — closest single reference allele and its r
"""
function neighbourhood_scores(R::AbstractMatrix, query_alleles::AbstractVector,
                              reference_alleles::AbstractVector, freq::AbstractVector;
                              k::Int, fisher::Bool=true)
    n_q, n_ref = size(R)
    n_q == length(query_alleles) ||
        error("neighbourhood_scores: R rows ($n_q) ≠ query alleles ($(length(query_alleles)))")
    n_ref == length(reference_alleles) == length(freq) ||
        error("neighbourhood_scores: R cols ($n_ref) must match reference alleles and freq")

    out = DataFrame(allele=String[], k=Int[], peak_r=Float64[], mean_r=Float64[],
                    nbhd_r=Float64[], weighted_nbhd_r=Float64[],
                    nearest_reference=String[], nearest_r=Float64[])
    for i in 1:n_q
        r = R[i, :]
        jbest = argmax(r)
        push!(out, (string(query_alleles[i]), k, maximum(r), mean(r),
                    topk_mean(r, k),
                    weighted_neighbourhood_r(r, freq, k; fisher=fisher),
                    string(reference_alleles[jbest]), r[jbest]))
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
# non-negative; a zero weight simply drops that allele from the geometry.
function _weight_columns(X::AbstractMatrix, w)
    w === nothing && return X
    length(w) == size(X, 2) ||
        error("cka: weight length $(length(w)) ≠ number of alleles $(size(X, 2))")
    X .* sqrt.(clamp.(Float64.(w), 0.0, Inf))'
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

"""
    PanelAlignment

Panel-level neighbourhood alignment — a single frequency-weighted CKA between a
query and reference panel with its permutation-null significance.

- `alignment`  — frequency-weighted linear CKA in [0, 1] (**primary panel metric**)
- `p`          — one-sided empirical p, `P(null ≥ observed)`
- `z`          — standardised effect `(observed − null mean) / null sd`;
                 comparable across panel pairs of different sizes
- `null_mean`, `null_sd`, `nperm` — the permutation-null summary
- `weighted`   — whether carrier frequencies were applied
- `n_query`, `n_reference`, `n_variants` — the aligned panel dimensions
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
end

Base.show(io::IO, pa::PanelAlignment) = print(io,
    "PanelAlignment(alignment=$(round(pa.alignment, digits=4)), " *
    "z=$(round(pa.z, digits=2)), p=$(pa.p), " *
    "$(pa.weighted ? "frequency-weighted" : "unweighted"), " *
    "$(pa.n_query)×$(pa.n_reference) alleles, $(pa.n_variants) variants)")

"""
    panel_alignment(Mq, Mr; wq=nothing, wr=nothing, nperm=1000,
                    rng=Random.default_rng(), weighted=false,
                    n_query=size(Mq,2), n_reference=size(Mr,2)) -> PanelAlignment

Neighbourhood alignment between two aligned escape-profile matrices, with a
permutation null. The null shuffles the reference's variant rows relative to the
query — breaking the variant alignment while preserving each panel's own
structure. A row permutation leaves each panel's own Gram matrix unchanged, so
the denominator is computed once and only the cross term is recomputed per
permutation. Returns a [`PanelAlignment`](@ref).
"""
function panel_alignment(Mq::AbstractMatrix, Mr::AbstractMatrix;
                         wq=nothing, wr=nothing, nperm::Int=1000,
                         rng=Random.default_rng(), weighted::Bool=false,
                         n_query::Int=size(Mq, 2), n_reference::Int=size(Mr, 2))
    size(Mq, 1) == size(Mr, 1) ||
        error("panel_alignment: query and reference have different variant counts")
    Xc = _weight_columns(column_center(Mq), wq)
    Yc = _weight_columns(column_center(Mr), wr)
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
    PanelAlignment(obs, p, z, μ, σ, nperm, weighted, n_query, n_reference, m)
end

"""
    panel_alignment(query::ProfileSet, reference::ProfileSet;
                    frequencies=nothing, nperm=1000, rng=Random.default_rng())

Convenience method: align `query` and `reference` on their shared variants and
compute the neighbourhood alignment. With `frequencies` (a `Dict` from
`load_frequencies`, keyed across both panels) both sides are weighted by carrier
frequency; without it the alignment is unweighted.
"""
function panel_alignment(query::ProfileSet, reference::ProfileSet;
                         frequencies=nothing, nperm::Int=1000,
                         rng=Random.default_rng())
    Mq, Mr = align_profiles(query, reference)
    weighted = frequencies !== nothing
    wq = weighted ? frequency_vector(query, frequencies)     : nothing
    wr = weighted ? frequency_vector(reference, frequencies) : nothing
    panel_alignment(Mq, Mr; wq=wq, wr=wr, nperm=nperm, rng=rng, weighted=weighted,
                    n_query=nalleles(query), n_reference=nalleles(reference))
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
                          n_reference=size(Mr,2)) -> AlignmentSensitivity

Compute the neighbourhood alignment on all shared variants and again on the
informative variants only (see [`informative_variants`](@ref)), returning both
with the fraction of variants retained and their difference. The masked run uses
the same weights, permutation count and RNG so the two are directly comparable.
"""
function alignment_sensitivity(Mq::AbstractMatrix, Mr::AbstractMatrix;
                               wq=nothing, wr=nothing, threshold::Real=0.0,
                               mode::Symbol=:either, nperm::Int=1000,
                               rng=Random.default_rng(), weighted::Bool=false,
                               n_query::Int=size(Mq, 2), n_reference::Int=size(Mr, 2))
    full = panel_alignment(Mq, Mr; wq=wq, wr=wr, nperm=nperm, rng=rng,
                           weighted=weighted, n_query=n_query, n_reference=n_reference)
    mask = informative_variants(Mq, Mr; threshold=threshold, mode=mode)
    keep = count(mask)
    keep >= 2 ||
        error("alignment_sensitivity: only $keep informative variant(s) at threshold=$threshold, mode=:$mode")
    masked = panel_alignment(Mq[mask, :], Mr[mask, :]; wq=wq, wr=wr, nperm=nperm,
                             rng=rng, weighted=weighted, n_query=n_query,
                             n_reference=n_reference)
    AlignmentSensitivity(full, masked, mode, Float64(threshold), size(Mq, 1), keep,
                         keep / size(Mq, 1), masked.alignment - full.alignment)
end

"""
    alignment_sensitivity(query::ProfileSet, reference::ProfileSet;
                          frequencies=nothing, threshold=0.0, mode=:either,
                          nperm=1000, rng=Random.default_rng())

Convenience method: align `query` and `reference` on their shared variants, then
run the zero-inflation sensitivity check. `frequencies` weights both panels by
carrier frequency, as in [`panel_alignment`](@ref).
"""
function alignment_sensitivity(query::ProfileSet, reference::ProfileSet;
                               frequencies=nothing, threshold::Real=0.0,
                               mode::Symbol=:either, nperm::Int=1000,
                               rng=Random.default_rng())
    Mq, Mr = align_profiles(query, reference)
    weighted = frequencies !== nothing
    wq = weighted ? frequency_vector(query, frequencies)     : nothing
    wr = weighted ? frequency_vector(reference, frequencies) : nothing
    alignment_sensitivity(Mq, Mr; wq=wq, wr=wr, threshold=threshold, mode=mode,
                          nperm=nperm, rng=rng, weighted=weighted,
                          n_query=nalleles(query), n_reference=nalleles(reference))
end
