# io.jl — loading CD8scape simulation output and panel/frequency definitions
#
# The core object is a `ProfileSet`: a set of MHC-I alleles, each described by a
# vector of CD8scape escape scores across a shared set of variants.

"""
    ProfileSet(keys, alleles, M)

An escape-profile matrix for a set of MHC-I alleles.

- `keys::DataFrame`   — one row per variant, holding the variant-identifying
                        columns (e.g. Gene, Locus, Mutation). Row order matches
                        the rows of `M`.
- `alleles::Vector{String}` — allele names, one per column of `M`.
- `M::Matrix{Float64}` — `nvariants × nalleles` matrix of escape scores.
                         Missing / non-binding entries are stored as 0.0.
"""
struct ProfileSet
    keys::DataFrame
    alleles::Vector{String}
    M::Matrix{Float64}
end

nvariants(ps::ProfileSet) = size(ps.M, 1)
nalleles(ps::ProfileSet)  = size(ps.M, 2)

Base.show(io::IO, ps::ProfileSet) =
    print(io, "ProfileSet($(nvariants(ps)) variants × $(nalleles(ps)) alleles)")

# ── helpers ─────────────────────────────────────────────────────────────────

# Strip a leading prefix from an allele column name (e.g. "BoLA_BoLA-1:009" → "BoLA-1:009").
_chop(s::AbstractString, prefix::AbstractString) =
    isempty(prefix) ? String(s) : String(chopprefix(s, prefix))

# Canonicalise an allele name for frequency lookup. NetMHCpan-style names drop
# the '*' that appears in standard HLA nomenclature (HLA-A*0101 → HLA-A0101);
# stripping it here lets a frequency table written either way match the profile
# columns. Harmless for BoLA / SLA names, which contain no '*'.
normalise_allele(a::AbstractString) = replace(String(a), "*" => "")

_stringify!(df::DataFrame, cols) = (for c in cols; df[!, c] = string.(df[!, c]); end; df)

# ── loaders ─────────────────────────────────────────────────────────────────

"""
    load_wide(path; key_cols, allele_prefix=nothing, allele_cols=nothing,
              strip_prefix="")

Load a *wide* CD8scape table: one row per variant, one column per allele.

`key_cols` names the variant-identifying columns. Allele columns are chosen by
`allele_prefix` (every column whose name starts with it) or given explicitly as
`allele_cols`. `strip_prefix` is removed from each chosen column name to recover
the bare allele name.

Example (combined_simulation.csv):
    load_wide(path; key_cols=["Gene","Locus","Mutation"],
              allele_prefix="BoLA_BoLA-", strip_prefix="BoLA_")
selects the 127 `BoLA_BoLA-*` columns (excluding the `BoLA_mean_ELBR_A` summary
column) and names them `BoLA-*`.
"""
function load_wide(path; key_cols, allele_prefix=nothing, allele_cols=nothing,
                   strip_prefix::AbstractString="")
    df = CSV.read(path, DataFrame)
    key_cols = String.(key_cols)
    _stringify!(df, key_cols)

    cols = if allele_cols !== nothing
        String.(allele_cols)
    elseif allele_prefix !== nothing
        filter(c -> startswith(c, allele_prefix), names(df))
    else
        error("load_wide: supply either `allele_prefix` or `allele_cols`")
    end
    isempty(cols) && error("load_wide: no allele columns matched")

    alleles = _chop.(cols, strip_prefix)
    M = Matrix{Float64}(coalesce.(df[:, cols], 0.0))
    ProfileSet(df[:, key_cols], alleles, M)
end

"""
    load_long(paths; key_cols, allele_col, value_col)

Load one or more *long* CD8scape tables (one row per (variant, allele) pair) and
combine them into a single variant × allele matrix. `paths` may be a single path
or a collection of paths; the rows of every file are pooled, then pivoted on
`allele_col` taking `value_col` as the score. This is how a panel split across
several CD8scape runs (e.g. one `per_allele_best_ranks.csv` per genome segment)
is combined. Duplicate (variant, allele) pairs are averaged; missing entries
become 0.

Example (a single long table):
    load_long("h5n1_simulation_sla.csv"; key_cols=["Gene","Locus","Mutation"],
              allele_col="MHC", value_col="log2_foldchange_BR")
"""
function load_long(paths; key_cols, allele_col, value_col)
    paths = paths isa AbstractString ? [paths] : collect(paths)
    isempty(paths) && error("load_long: no input files given")
    key_cols = String.(key_cols)
    allele_col = String(allele_col); value_col = String(value_col)
    need = vcat(key_cols, allele_col, value_col)

    parts = DataFrame[]
    for p in paths
        df = CSV.read(p, DataFrame)
        absent = setdiff(need, names(df))
        isempty(absent) || error("load_long: $p is missing column(s) $(absent)")
        push!(parts, select(df, need))
    end
    combined = length(parts) == 1 ? parts[1] : reduce(vcat, parts)

    _stringify!(combined, key_cols)
    combined[!, allele_col] = string.(combined[!, allele_col])

    # average any duplicate (variant, allele) entries so unstack is unambiguous
    agg = combine(groupby(combined, vcat(key_cols, allele_col)),
                  value_col => (v -> mean(skipmissing(v))) => value_col)
    w = unstack(agg, key_cols, allele_col, value_col)

    alleles = String.(setdiff(names(w), key_cols))
    sort!(alleles)
    M = Matrix{Float64}(coalesce.(w[:, alleles], 0.0))
    ProfileSet(w[:, key_cols], alleles, M)
end

# CD8scape `per_allele_best_ranks.csv` convention: variants keyed on
# (Frame, Locus, Mutation), allele in `MHC`, escape profile in `log2_foldchange_BR`.
const CD8SCAPE_KEY_COLS = ["Frame", "Locus", "Mutation"]

"""
    load_cd8scape(paths; key_cols=["Frame","Locus","Mutation"],
                  allele_col="MHC", value_col="log2_foldchange_BR")

Convenience loader for raw CD8scape `per_allele_best_ranks.csv` output. Accepts
one path or a list of them (combined into one panel), with CD8scape's column
defaults — no pre-processing of the files required. `value_col` defaults to the
per-allele escape metric `log2_foldchange_BR`.
"""
load_cd8scape(paths; key_cols=CD8SCAPE_KEY_COLS, allele_col="MHC",
              value_col="log2_foldchange_BR") =
    load_long(paths; key_cols=key_cols, allele_col=allele_col, value_col=value_col)

"""
    load_frequencies(path; allele_col="Allele", freq_col="Frequency")

Load carrier / population frequencies keyed by allele name. Returns a
`Dict{String,Float64}` keyed on the canonicalised allele name (see
`normalise_allele`).
"""
function load_frequencies(path; allele_col="Allele", freq_col="Frequency")
    df = CSV.read(path, DataFrame)
    Dict(normalise_allele(a) => Float64(f)
         for (a, f) in zip(df[!, String(allele_col)], df[!, String(freq_col)]))
end

"""
    frequency_vector(ps::ProfileSet, freqs::AbstractDict) -> Vector{Float64}

Align a frequency dictionary to the alleles of `ps`. Alleles with no entry get
0.0. If `freqs` is `nothing`, returns uniform weights (all 1.0), i.e. an
unweighted analysis.
"""
frequency_vector(ps::ProfileSet, ::Nothing) = ones(nalleles(ps))
frequency_vector(ps::ProfileSet, freqs::AbstractDict) =
    [get(freqs, normalise_allele(a), 0.0) for a in ps.alleles]

"""
    read_allele_list(path; allele_col="Allele") -> Vector{String}

Read a panel definition CSV and return its unique allele names.
"""
function read_allele_list(path; allele_col="Allele")
    df = CSV.read(path, DataFrame)
    unique(string.(df[!, String(allele_col)]))
end

"""
    subset_alleles(ps::ProfileSet, wanted) -> ProfileSet

Restrict `ps` to the alleles in `wanted` (in the order they appear in `ps`).
Alleles in `wanted` not present in `ps` are dropped with a warning.
"""
function subset_alleles(ps::ProfileSet, wanted)
    wset = Set(string.(wanted))
    keep = findall(a -> a in wset, ps.alleles)
    missing_alleles = setdiff(wset, Set(ps.alleles))
    isempty(missing_alleles) ||
        @warn "subset_alleles: $(length(missing_alleles)) requested allele(s) absent from profiles" missing_alleles
    ProfileSet(ps.keys, ps.alleles[keep], ps.M[:, keep])
end

"""
    align_profiles(query::ProfileSet, reference::ProfileSet) -> (Mq, Mr)

Inner-join `query` and `reference` on their (identical) key columns and return
their score matrices restricted and ordered to the shared variants. When both
sets come from the same table this is a row-for-row pass-through; when they come
from different CD8scape runs it matches variants across them.
"""
function align_profiles(query::ProfileSet, reference::ProfileSet)
    kc = names(query.keys)
    names(reference.keys) == kc ||
        error("align_profiles: key columns differ ($(kc) vs $(names(reference.keys)))")

    qk = copy(query.keys);     qk.__qrow = 1:nrow(qk)
    rk = copy(reference.keys); rk.__rrow = 1:nrow(rk)
    j = innerjoin(qk, rk, on=kc)
    nrow(j) == 0 && error("align_profiles: query and reference share no variants")

    return query.M[j.__qrow, :], reference.M[j.__rrow, :]
end
