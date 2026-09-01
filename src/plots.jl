# plots.jl — figure ports for the MHC-I neighbourhood analysis (CairoMakie).
#
# A standalone module so the core `MHC1Neighbourhoods` module stays light
# (CSV / DataFrames / Statistics only). Load it separately when you want
# figures:
#
#     include("src/plots.jl"); using .NeighbourhoodPlots
#
# Every function returns a CairoMakie `Figure`; `save_all_figures` writes the
# full set to a directory. Colours follow the project-wide viridis convention.

module NeighbourhoodPlots

using CairoMakie
using Statistics
using KernelDensity
using MultivariateStats
using UMAP
using Random
using ColorSchemes
using DataFrames

export plot_k_calibration, plot_k_sd, plot_mean_peak, plot_weighted_unweighted,
       plot_ranked, plot_profile_densities, umap_embedding,
       plot_umap_species, plot_umap_scored, save_all_figures

CairoMakie.activate!(type="png")

# viridis anchors (project convention): purple / teal / yellow
const PURPLE = ColorSchemes.viridis[0.0]
const TEAL   = ColorSchemes.viridis[0.5]
const YELLOW = ColorSchemes.viridis[1.0]

# ── k-selection figures (Supp Fig 2C / 2D) ──────────────────────────────────

"""
    plot_k_calibration(calibration; k_star=nothing)

cor(top-k, peak r) and cor(top-k, mean r) as a function of k. The crossover is
where they meet; `k_star` (if given) is marked.
"""
function plot_k_calibration(cal::DataFrame; k_star=nothing)
    fig = Figure(size=(720, 460))
    ax = Axis(fig[1, 1]; xlabel="k", ylabel="Pearson r",
              title="Neighbourhood-size selection")
    lines!(ax, cal.k, cal.cor_peak; color=PURPLE, linewidth=2.5, label="cor(top-k, peak r)")
    lines!(ax, cal.k, cal.cor_mean; color=TEAL, linewidth=2.5, label="cor(top-k, mean r)")
    k_star === nothing ||
        vlines!(ax, [k_star]; color=:gray40, linestyle=:dash, label="k* = $k_star")
    axislegend(ax; position=:rc)
    fig
end

"""
    plot_k_sd(calibration)

Spread (SD across query alleles) of the top-k neighbourhood score vs k.
"""
function plot_k_sd(cal::DataFrame)
    fig = Figure(size=(720, 460))
    ax = Axis(fig[1, 1]; xlabel="k", ylabel="SD of neighbourhood r",
              title="Discriminability vs k")
    lines!(ax, cal.k, cal.sd_topk; color=TEAL, linewidth=2.5)
    fig
end

# ── similarity scatter (Fig 1C) ─────────────────────────────────────────────

"""
    plot_mean_peak(scores)

Each query allele's mean r (x) vs peak r (y), coloured by top-k neighbourhood r.
"""
function plot_mean_peak(scores::DataFrame)
    fig = Figure(size=(760, 560))
    ax = Axis(fig[1, 1]; title="Breadth vs depth of similarity",
              xlabel="mean Pearson r (query vs all reference)",
              ylabel="peak Pearson r (query vs best reference)")
    sc = scatter!(ax, scores.mean_r, scores.peak_r; color=scores.nbhd_r,
                  colormap=:viridis, markersize=12, strokewidth=0.3, strokecolor=:white)
    Colorbar(fig[1, 2], sc; label="neighbourhood r")
    fig
end

# ── weighted vs unweighted (Fig 1D) ─────────────────────────────────────────

"""
    plot_weighted_unweighted(scores)

Frequency-weighted vs unweighted neighbourhood r, coloured by peak r, with the
line of identity. Points above the line are up-weighted by allele frequency.
"""
function plot_weighted_unweighted(scores::DataFrame)
    lo = min(minimum(scores.nbhd_r), minimum(scores.weighted_nbhd_r))
    hi = max(maximum(scores.nbhd_r), maximum(scores.weighted_nbhd_r))
    pad = 0.02 * (hi - lo)
    fig = Figure(size=(760, 600))
    ax = Axis(fig[1, 1]; title="Effect of frequency weighting",
              xlabel="unweighted neighbourhood r", ylabel="weighted neighbourhood r")
    lines!(ax, [lo - pad, hi + pad], [lo - pad, hi + pad];
           color=:gray60, linestyle=:dash)
    sc = scatter!(ax, scores.nbhd_r, scores.weighted_nbhd_r; color=scores.peak_r,
                  colormap=:viridis, markersize=12, strokewidth=0.3, strokecolor=:white)
    Colorbar(fig[1, 2], sc; label="peak r")
    fig
end

# ── ranked bars (Fig 2A) ────────────────────────────────────────────────────

"""
    plot_ranked(scores; n=15)

Horizontal bar chart of the `n` query alleles with the highest weighted
neighbourhood r, coloured on the viridis scale.
"""
function plot_ranked(scores::DataFrame; n::Int=15)
    n = min(n, nrow(scores))
    top = first(scores, n)
    ys = collect(n:-1:1)                    # rank 1 at the top
    fig = Figure(size=(760, 40n + 140))
    ax = Axis(fig[1, 1]; title="Top $n alleles by weighted neighbourhood r",
              xlabel="weighted neighbourhood r", yticks=(ys, top.allele))
    barplot!(ax, ys, top.weighted_nbhd_r; direction=:x,
             color=top.weighted_nbhd_r, colormap=:viridis)
    Colorbar(fig[1, 2]; colormap=:viridis,
             limits=(minimum(top.weighted_nbhd_r), maximum(top.weighted_nbhd_r)),
             label="weighted r")
    fig
end

# ── simulation-value densities (Supp Fig 2A / 2B) ───────────────────────────

"""
    plot_profile_densities(M, alleles=nothing; color=PURPLE, label="",
                           clip=0.001, drop_nonbinding=true, min_points=5)

Overlaid kernel-density estimates of the escape-score distribution for each
allele (column of `M`). Non-binding entries are stored as exact zeros; by
default (`drop_nonbinding`) they are excluded, so the curves show the
distribution of *actual* escape scores rather than a spike at 0. `clip` trims
extreme tails for a readable x-range; alleles with fewer than `min_points`
scores are skipped.
"""
function plot_profile_densities(M::AbstractMatrix, alleles=nothing;
                                color=PURPLE, label::AbstractString="",
                                clip=0.001, drop_nonbinding::Bool=true, min_points::Int=5)
    keepcol(v) = drop_nonbinding ? filter(!iszero, v) : collect(v)
    pooled = keepcol(vec(M))
    isempty(pooled) && (pooled = vec(M))
    lo, hi = quantile(pooled, clip), quantile(pooled, 1 - clip)
    fig = Figure(size=(780, 440))
    ax = Axis(fig[1, 1]; xlabel="simulation value", ylabel="density",
              title=label, limits=((lo, hi), nothing))
    for j in 1:size(M, 2)
        col = keepcol(@view M[:, j])
        (length(col) < min_points || all(==(first(col)), col)) && continue
        k = kde(col)
        lines!(ax, k.x, k.density; color=(color, 0.12), linewidth=0.8)
    end
    fig
end

# ── UMAP embedding (Fig 1A / 1B / 2B) ───────────────────────────────────────

"""
    umap_embedding(Mq, Mr; n_neighbors=15, min_dist=0.1, n_pca=50, seed=1320) -> (emb, is_query)

Project the query and reference escape profiles into a shared 2-D UMAP layout.
`Mq` and `Mr` are variant×allele matrices aligned on the same variants. The
allele×variant matrix is z-scored per variant, reduced to `n_pca` principal
components, then embedded with UMAP.

Returns `emb` (2 × n_alleles, query alleles first) and a `Bool` vector flagging
the query alleles.

Note: UMAP is a stochastic visualisation only; layouts differ between
implementations (this uses UMAP.jl, not R's uwot) and none of the neighbourhood
statistics depend on it.
"""
function umap_embedding(Mq::AbstractMatrix, Mr::AbstractMatrix;
                        n_neighbors::Int=15, min_dist::Real=0.1,
                        n_pca::Int=50, seed::Int=1320)
    X = permutedims(hcat(Mq, Mr))                     # alleles × variants
    Xz = (X .- mean(X, dims=1)) ./ (std(X, dims=1) .+ eps())
    feat = permutedims(Xz)                            # variants × alleles (features × obs)
    maxout = min(n_pca, size(feat, 1) - 1, size(feat, 2) - 1)
    pca = MultivariateStats.fit(PCA, feat; maxoutdim=maxout)
    pcs = MultivariateStats.predict(pca, feat)        # (n_pca × n_alleles)
    Random.seed!(seed)
    emb = UMAP.fit(pcs, 2; n_neighbors=n_neighbors, min_dist=min_dist).embedding  # 2 × n_alleles
    is_query = vcat(trues(size(Mq, 2)), falses(size(Mr, 2)))
    return emb, is_query
end

"""
    plot_umap_species(emb, is_query; query_label="query", reference_label="reference")

UMAP scatter with query (purple) and reference (yellow) alleles (Fig 1A).
"""
function plot_umap_species(emb::AbstractMatrix, is_query::AbstractVector;
                           query_label="query", reference_label="reference")
    fig = Figure(size=(720, 620))
    ax = Axis(fig[1, 1]; xlabel="UMAP 1", ylabel="UMAP 2",
              title="Escape-profile binding space")
    scatter!(ax, emb[1, .!is_query], emb[2, .!is_query]; color=YELLOW,
             markersize=9, strokewidth=0.3, strokecolor=:white, label=reference_label)
    scatter!(ax, emb[1, is_query], emb[2, is_query]; color=PURPLE,
             markersize=9, strokewidth=0.3, strokecolor=:white, label=query_label)
    axislegend(ax)
    fig
end

"""
    plot_umap_scored(emb, is_query, query_alleles, weighted_r; label_top=15, contours=true)

UMAP with reference alleles in grey (+ density contours) and query alleles
coloured by weighted neighbourhood r; the `label_top` highest-scoring query
alleles are labelled (Fig 2B). `weighted_r` must be aligned to `query_alleles`,
which must be in the same order as the query columns of `emb`.
"""
function plot_umap_scored(emb::AbstractMatrix, is_query::AbstractVector,
                          query_alleles::AbstractVector, weighted_r::AbstractVector;
                          label_top::Int=15, contours::Bool=true)
    rx, ry = emb[1, .!is_query], emb[2, .!is_query]
    qx, qy = emb[1, is_query], emb[2, is_query]
    fig = Figure(size=(760, 620))
    ax = Axis(fig[1, 1]; xlabel="UMAP 1", ylabel="UMAP 2",
              title="Query alleles by weighted neighbourhood r")
    if contours && length(rx) > 5
        k2 = kde((rx, ry))
        contour!(ax, k2.x, k2.y, k2.density; color=:gray70, linewidth=1)
    end
    scatter!(ax, rx, ry; color=:gray75, markersize=6)
    sc = scatter!(ax, qx, qy; color=collect(weighted_r), colormap=:viridis,
                  markersize=11, strokewidth=0.3, strokecolor=:white)
    Colorbar(fig[1, 2], sc; label="weighted r")
    # Label the top-scoring query alleles, skipping any whose anchor point sits
    # too close to an already-placed label so text never stacks/garbles. Labels
    # flip side near the right edge to stay inside the axes. No glyph stroke.
    n = min(label_top, length(weighted_r))
    order = partialsortperm(collect(weighted_r), 1:n, rev=true)
    xr = maximum(qx) - minimum(qx); yr = maximum(qy) - minimum(qy)
    dmin = 0.045 * max(xr, yr)
    xmid = (minimum(qx) + maximum(qx)) / 2
    placed = NTuple{2,Float64}[]
    for i in order
        px, py = qx[i], qy[i]
        any(p -> hypot(px - p[1], py - p[2]) < dmin, placed) && continue
        push!(placed, (px, py))
        right = px > xmid
        text!(ax, px, py; text=String(query_alleles[i]), fontsize=8, color=:black,
              align=(right ? :right : :left, :bottom), offset=(right ? -6 : 6, 4))
    end
    fig
end

# ── assemble everything ─────────────────────────────────────────────────────

"""
    save_all_figures(res, query_profiles, reference_profiles; outdir,
                     query_label="query", reference_label="reference",
                     top_n=15, umap=true, fmt="png", seed=1320)

Write the full figure set for a `Neighbourhood` result `res` (from
`MHC1Neighbourhoods.analyse`) to `outdir`. `query_profiles`/`reference_profiles`
are the `ProfileSet`s passed to `analyse` (used for the density and UMAP panels).
Set `umap=false` to skip the UMAP panels. Returns the vector of written paths.
"""
function save_all_figures(res, query_profiles, reference_profiles; outdir::AbstractString,
                          query_label="query", reference_label="reference",
                          top_n::Int=15, umap::Bool=true, fmt::AbstractString="png",
                          seed::Int=1320)
    mkpath(outdir)
    written = String[]
    _write(name, fig) = (p = joinpath(outdir, name * "." * fmt); save(p, fig); push!(written, p))

    _write("k_calibration",          plot_k_calibration(res.calibration; k_star=res.k))
    _write("k_sd",                   plot_k_sd(res.calibration))
    _write("mean_vs_peak",           plot_mean_peak(res.scores))
    _write("weighted_vs_unweighted", plot_weighted_unweighted(res.scores))
    _write("ranked_top$(top_n)",     plot_ranked(res.scores; n=top_n))
    _write("densities_$(query_label)",
           plot_profile_densities(query_profiles.M; color=PURPLE, label=query_label))
    _write("densities_$(reference_label)",
           plot_profile_densities(reference_profiles.M; color=YELLOW, label=reference_label))

    if umap
        Mq, Mr = _aligned_matrices(query_profiles, reference_profiles)
        emb, is_q = umap_embedding(Mq, Mr; seed=seed)
        _write("umap_species",
               plot_umap_species(emb, is_q; query_label=query_label, reference_label=reference_label))
        wmap = Dict(res.scores.allele .=> res.scores.weighted_nbhd_r)
        wr = [get(wmap, a, NaN) for a in query_profiles.alleles]
        _write("umap_scored",
               plot_umap_scored(emb, is_q, query_profiles.alleles, wr; label_top=top_n))
    end
    return written
end

# local re-implementation of align (avoids depending on the parent module)
function _aligned_matrices(q, r)
    kc = names(q.keys)
    qk = copy(q.keys); qk.__qrow = 1:nrow(qk)
    rk = copy(r.keys); rk.__rrow = 1:nrow(rk)
    j = innerjoin(qk, rk, on=kc)
    return q.M[j.__qrow, :], r.M[j.__rrow, :]
end

end # module
