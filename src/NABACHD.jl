"""
    NABACHD — Neighbourhood Analysis of Binding-escape Against Class-I HLA Data

A framework for quantifying the meaningfulness of MHC-I binding escape from one
allele panel relative to another, from CD8scape escape-profile output. The query
and reference panels are both configurable, so the method applies to any pair of
panels — a non-human panel against the human HLA repertoire, or one human panel
against another. (NABACHD ≈ Scottish Gaelic *nàbachd*, "neighbourhood".)

For each allele in a *query* panel, its per-variant escape profile is correlated
against every allele in a (larger, optionally frequency-weighted) *reference*
panel. A neighbourhood size k is chosen by a crossover criterion, and the
frequency-weighted top-k neighbourhood r summarises how closely the query
allele's escape landscape is mirrored by the reference repertoire.

The query and reference panels are both configurable, so the method applies to
any pair of panels handled by netMHCpan / CD8scape — not only BoLA vs HLA.

Typical use (see `bin/run.jl` for a config-driven CLI):

```julia
using NABACHD
res = analyse(
    load_wide("combined_simulation.csv";
              key_cols=["Gene","Locus","Mutation"],
              allele_prefix="BoLA_BoLA-", strip_prefix="BoLA_"),      # query
    load_wide("combined_simulation.csv";
              key_cols=["Gene","Locus","Mutation"],
              allele_prefix="HLA_HLA-", strip_prefix="HLA_");         # reference
    frequencies = load_frequencies("supertype_panel.csv"),
    k = :auto)
res.scores      # ranked DataFrame
res.k           # selected neighbourhood size
res.calibration # k-sweep table
```
"""
module NABACHD

using CSV
using DataFrames
using Statistics

include("io.jl")
include("neighbourhood.jl")

export ProfileSet, nvariants, nalleles,
       load_wide, load_long, load_cd8scape, load_frequencies, frequency_vector,
       read_allele_list, subset_alleles, align_profiles,
       correlation_matrix, topk_mean, weighted_neighbourhood_r,
       default_k_sequence, calibrate_k, find_k_star, neighbourhood_scores,
       fisher_z, fisher_zinv,
       Neighbourhood, analyse

"""
    Neighbourhood

Result of an `analyse` run:

- `scores::DataFrame`       — per query allele, ranked by weighted neighbourhood r
- `calibration::DataFrame`  — the k-sweep table (`k, cor_peak, cor_mean, sd_topk`)
- `k::Int`                  — neighbourhood size used
- `R::Matrix{Float64}`      — query × reference correlation matrix
- `query_alleles`, `reference_alleles`, `frequencies` — the aligned inputs
"""
struct Neighbourhood
    scores::DataFrame
    calibration::DataFrame
    k::Int
    R::Matrix{Float64}
    query_alleles::Vector{String}
    reference_alleles::Vector{String}
    frequencies::Vector{Float64}
end

Base.show(io::IO, n::Neighbourhood) = print(io,
    "Neighbourhood(k=$(n.k), $(length(n.query_alleles)) query × " *
    "$(length(n.reference_alleles)) reference alleles)")

"""
    analyse(query::ProfileSet, reference::ProfileSet;
            frequencies=nothing, k=:auto, fisher=true, k_seq=nothing) -> Neighbourhood

Run the full neighbourhood analysis.

- `frequencies` — a `Dict` from `load_frequencies`, or `nothing` for an
                  unweighted analysis (uniform weights).
- `k`           — `:auto` to select k* by the crossover criterion, or an integer
                  to fix the neighbourhood size.
- `fisher`      — Fisher z-average the weighted metric (default `true`).
- `k_seq`       — optional explicit k sweep for calibration.
"""
function analyse(query::ProfileSet, reference::ProfileSet;
                 frequencies=nothing, k=:auto, fisher::Bool=true, k_seq=nothing)
    Mq, Mr = align_profiles(query, reference)
    # warn about alleles with no binding variation across the shared variants:
    # their correlations are undefined and will be treated as 0.
    qbad = count(j -> std(view(Mq, :, j)) == 0, axes(Mq, 2))
    rbad = count(j -> std(view(Mr, :, j)) == 0, axes(Mr, 2))
    (qbad > 0 || rbad > 0) && @warn "Alleles with no binding variation across the shared variants; their correlations are treated as 0" query_degenerate = qbad reference_degenerate = rbad
    R = correlation_matrix(Mq, Mr)

    ks = k_seq === nothing ? default_k_sequence(size(R, 2)) : k_seq
    if k === :auto
        k_used, calibration = find_k_star(R; k_seq=ks)
    else
        k_used = Int(k)
        calibration = calibrate_k(R; k_seq=ks)
    end

    freq = frequency_vector(reference, frequencies)
    scores = neighbourhood_scores(R, query.alleles, reference.alleles, freq;
                                  k=k_used, fisher=fisher)
    Neighbourhood(scores, calibration, k_used, R,
                  copy(query.alleles), copy(reference.alleles), freq)
end

end # module
