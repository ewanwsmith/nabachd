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
    cor(Mq, Mr)
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
