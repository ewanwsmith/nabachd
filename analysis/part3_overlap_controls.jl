#!/usr/bin/env julia
#
# Part 3 overlap controls (2026-11-18). Answers the two runs
# `claude/part3-overlap-note.md` §"What would actually settle it" asks for.
#
#   task2  the shared-dropped control with UNIFORM weights. The weighted
#          shared_dropped row is degenerate — dropping the 406 shared alleles
#          leaves a UK panel of participation ratio 1.1 out of 376, because the
#          alleles both registries hold are the common ones and the UK-only
#          remainder is the rare tail. Panel MEMBERSHIP still comes from the
#          frequency files (782 UK, 2078 US alleles with positive frequency);
#          only the weights become 1, which removes the weight concentration
#          while keeping the mechanical-inflation question intact. The ceiling
#          is drawn in mode :uniform to match.
#          → part3_uk_vs_us_alignment_unweighted.csv
#
#   task3  the retained-fraction curve, replacing the all-or-nothing switch.
#          For each f, keep a random fraction f of the 406 shared alleles in
#          BOTH panels and drop the rest from BOTH, then compute the WEIGHTED
#          statistic (ceiling weight-matched to the query, as part3 does).
#          Interior f are averaged over several draws; bootstrap CIs are run at
#          f = 0 and f = 1 only, since the bootstrap resamples variants and so
#          cannot reuse the Gram.
#          → part3_uk_vs_us_overlap_curve.csv
#
# NEITHER TASK WRITES part3_uk_vs_us_alignment.csv. That file is the output of a
# 7 h 47 m run; a copy is kept beside it as .20261117. part3() overwrites it, so
# this script never calls part3().
#
# Usage, from the repo root:
#   julia --project=. analysis/part3_overlap_controls.jl task2
#   julia --project=. analysis/part3_overlap_controls.jl task3
#   julia --project=. analysis/part3_overlap_controls.jl all
#
# Env:
#   NABACHD_P3_NBOOT   bootstrap resamples (default 200; 50 while developing)
#   NABACHD_P3_CEIL    ceiling draws       (default 100)
#   NABACHD_P3_DRAWS   draws per interior f in task3 (default 5)
#   NABACHD_P3_CURVE_WEIGHTS  freq (default) | uniform — task3 weighting. uniform
#                      writes part3_uk_vs_us_overlap_curve_unweighted.csv instead

include(joinpath(@__DIR__, "writeup_runs.jl"))
using DataFrames, CSV, Random, Statistics

const P3_DRAWS = parse(Int, get(ENV, "NABACHD_P3_DRAWS", "5"))
const FRACS    = [0.0, 0.25, 0.5, 0.75, 1.0]

# ── shared load ──────────────────────────────────────────────────────────────
# One 6.6 GB load serves every row. Returns the aligned profile matrices, the
# normalised allele names (so membership tests are name-comparable across
# panels) and the full frequency vectors.
function _load_p3()
    @info "loading UK/US registry panels (SARS-CoV-2)"
    uk = load_cd8scape(require(joinpath(HLAN, "UK", "per_allele_best_ranks_simulated.csv")))
    us = load_cd8scape(require(joinpath(HLAN, "US", "per_allele_best_ranks_simulated.csv")))
    # One dictionary per panel, never merged — see the note in part3(). A merged
    # dictionary weights the UK panel's 406 shared alleles by US frequencies.
    freq_uk = load_frequencies(joinpath(HLAN, "UK", "supertype_panel_squished.csv");
                               on_duplicate = :error)
    freq_us = load_frequencies(joinpath(HLAN, "US", "supertype_panel_validated.csv");
                               on_duplicate = :error)
    Mq, Mr = align_profiles(uk, us)
    nmz(x) = nabachd.normalise_allele(x)
    uk_names, us_names = nmz.(uk.alleles), nmz.(us.alleles)
    shared = intersect(Set(uk_names), Set(us_names))
    @info "  UK $(nalleles(uk)) × US $(nalleles(us)) catalogue alleles, " *
          "$(size(Mq, 1)) shared variants, $(length(shared)) shared alleles"
    (Mq = Mq, Mr = Mr, uk_names = uk_names, us_names = us_names, shared = shared,
     wq_all = frequency_vector(uk, freq_uk), wr_all = frequency_vector(us, freq_us))
end

# Effective dimensionality of a panel: the participation ratio of its weighted
# Gram's eigenvalues, (Σλ)²/Σλ². For a symmetric PSD Gram that is exactly
# (tr G)² / ‖G‖_F², so no eigendecomposition is needed — the identity is exact,
# not an approximation, and avoids decomposing a 2078 × 2078 matrix per draw.
function _effdim(X::AbstractMatrix, w)
    Xw = X .* sqrt.(clamp.(Float64.(w), 0.0, Inf))'
    G  = Xw' * Xw
    t  = sum(i -> G[i, i], 1:size(G, 1))
    f2 = sum(abs2, G)
    f2 == 0 ? 0.0 : t^2 / f2
end

# _ceiling now errors rather than clamping, and the construction needs the query
# panel to be the smaller one. Check it ourselves first so the message names the
# row rather than coming out of a helper three frames down.
function _check_sizes(nq, nr, label)
    1 <= nq < nr || error("$label: need 1 <= n_query_eff < n_ref_eff " *
                          "(the ceiling splits the reference into a query-sized " *
                          "part and a remainder); got n_query_eff = $nq, " *
                          "n_ref_eff = $nr")
end

# One row's worth of statistics for a given pair of allele index sets.
#   wmode  :uniform → unit weights and a :uniform ceiling (task 2)
#          :freq    → the frequency weights and a query-weight-matched ceiling
#                     (task 3, matching part3's primary figure)
#   boot   run the bootstrap CI (expensive: it resamples variants, so the Gram
#          cannot be reused — roughly 8 s per resample)
function _stats(d, kq, kr, wmode::Symbol, label; boot::Bool)
    wq = wmode === :uniform ? ones(length(kq)) : d.wq_all[kq]
    wr = wmode === :uniform ? ones(length(kr)) : d.wr_all[kr]
    nq, nr = length(kq), length(kr)
    _check_sizes(nq, nr, label)
    rng() = MersenneTwister(SEED)

    # single-centred first, then freed — both pairs at once is ~2.7 GB of copies
    Sq, Sr = _prepare(d.Mq, kq, wq, :single), _prepare(d.Mr, kr, wr, :single)
    a_sw = cka(Sq, Sr; wq = wq, wr = wr)
    Sq = Sr = nothing; GC.gc()

    Dq, Dr = _prepare(d.Mq, kq, wq, :double), _prepare(d.Mr, kr, wr, :double)
    a_dw = cka(Dq, Dr; wq = wq, wr = wr)
    ed_q, ed_r = _effdim(Dq, wq), _effdim(Dr, wr)
    cmode = wmode === :uniform ? :uniform : wq
    c = _ceiling(Dr, wr, nq, cmode; rng = rng(), reps = P3_CEIL)
    ci = boot ? _boot_ci(Dq, Dr, wq, wr; nboot = P3_NBOOT, rng = rng()) :
                (lo = missing, hi = missing, sd = missing)
    Dq = Dr = nothing; GC.gc()

    (n_variants = size(d.Mq, 1), n_query_eff = nq, n_ref_eff = nr,
     pr_query = _pr(wq), pr_ref = _pr(wr),
     pr_query_freq = _pr(d.wq_all[kq]), pr_ref_freq = _pr(d.wr_all[kr]),
     effdim_query = ed_q, effdim_ref = ed_r,
     align_single = a_sw, align_double = a_dw,
     lo = ci.lo, hi = ci.hi,
     ceiling = c.mean, ceiling_sd = c.sd, ceiling_reps = P3_CEIL,
     ceiling_sem = c.sd / sqrt(P3_CEIL),
     ceiling_mode = cmode isa Symbol ? String(cmode) : "query_matched",
     frac_ceiling = a_dw / c.mean,
     frac_ceiling_lo = boot ? ci.lo / c.mean : missing,
     frac_ceiling_hi = boot ? ci.hi / c.mean : missing)
end

# ── Task 2 — unweighted shared-dropped control ───────────────────────────────
function task2(d = _load_p3())
    @info "Task 2 — shared-dropped control with uniform weights"
    rows = DataFrame()
    for (tag, drop) in (("all", false), ("shared_dropped", true))
        kq = findall(i -> d.wq_all[i] > 0 && !(drop && d.uk_names[i] in d.shared),
                     eachindex(d.wq_all))
        kr = findall(i -> d.wr_all[i] > 0 && !(drop && d.us_names[i] in d.shared),
                     eachindex(d.wr_all))
        (isempty(kq) || isempty(kr)) &&
            error("task2: $tag left one side with no allele")
        ns = count(i -> d.uk_names[i] in d.shared, kq)
        s  = _stats(d, kq, kr, :uniform, "task2/$tag"; boot = true)
        push!(rows, merge((variant = tag, weights = "uniform", n_shared = ns,
                           frac_shared_query = ns / s.n_query_eff,
                           frac_shared_reference =
                               count(i -> d.us_names[i] in d.shared, kr) /
                               s.n_ref_eff),
                          s); promote = true)
        @info "  $tag: UK($(s.n_query_eff))→US($(s.n_ref_eff)), $ns shared: " *
              "single=$(round(s.align_single, digits = 4)) " *
              "double=$(round(s.align_double, digits = 4)) " *
              "ceiling=$(round(s.ceiling, digits = 4))±$(round(s.ceiling_sd, digits = 3)) " *
              "→ $(round(100 * s.frac_ceiling, digits = 1))% of ceiling " *
              "(PR $(round(s.pr_query, digits = 1)); " *
              "PR under frequencies would be $(round(s.pr_query_freq, digits = 1)))"
    end
    mkpath(RESULTS)
    f = joinpath(RESULTS, "part3_uk_vs_us_alignment_unweighted.csv")
    CSV.write(f, rows); @info "  → $f  ($(nrow(rows)) rows)"
    rows
end

# ── Task 3 — retained-fraction curve ─────────────────────────────────────────
# NABACHD_P3_CURVE_WEIGHTS=uniform reruns the curve with unit weights. Worth
# having: under frequency weights each interior f is a lottery over weight mass
# (which of the 406 shared alleles a draw keeps decides whether the common
# heavyweights are in), so the across-draw sd of frac_ceiling (0.11-0.24) is
# larger than the whole f=0 to f=1 trend (0.17) and the ceiling itself is barely
# estimable (sd ~0.10). Uniform weights remove the lottery exactly as they do in
# task 2, where the ceiling's sd falls to 0.004-0.008.
function task3(d = _load_p3())
    wmode = Symbol(get(ENV, "NABACHD_P3_CURVE_WEIGHTS", "freq"))
    wmode in (:freq, :uniform) ||
        error("NABACHD_P3_CURVE_WEIGHTS must be freq or uniform; got $wmode")
    @info "Task 3 — retained-fraction curve over the shared block (weights: $wmode)"
    shared_list = sort(collect(d.shared))
    nsh = length(shared_list)
    out = DataFrame()
    for (fi, f) in enumerate(FRACS)
        endpoint = (f == 0.0 || f == 1.0)
        ndraw = endpoint ? 1 : P3_DRAWS      # f = 0 and f = 1 are deterministic
        per = DataFrame()
        for draw in 1:ndraw
            nkeep = round(Int, f * nsh)
            keep  = Set(shared_list[randperm(MersenneTwister(SEED + 1000 * fi + draw),
                                             nsh)[1:nkeep]])
            dropped(name) = (name in d.shared) && !(name in keep)
            kq = findall(i -> d.wq_all[i] > 0 && !dropped(d.uk_names[i]),
                         eachindex(d.wq_all))
            kr = findall(i -> d.wr_all[i] > 0 && !dropped(d.us_names[i]),
                         eachindex(d.wr_all))
            ns = count(i -> d.uk_names[i] in keep, kq)
            s  = _stats(d, kq, kr, wmode, "task3/f=$f draw $draw"; boot = endpoint)
            push!(per, merge((f = f, draw = draw, n_shared = ns), s); promote = true)
            @info "  f=$f draw $draw: UK($(s.n_query_eff))→US($(s.n_ref_eff)), " *
                  "$ns shared: double=$(round(s.align_double, digits = 4)) " *
                  "ceiling=$(round(s.ceiling, digits = 4)) " *
                  "→ $(round(100 * s.frac_ceiling, digits = 1))% " *
                  "(PR_q $(round(s.pr_query, digits = 2)), " *
                  "eff.dim $(round(s.effdim_query, digits = 2)) / " *
                  "$(round(s.effdim_ref, digits = 2)))"
        end
        # one row per f: point estimates averaged over draws, with the spread
        # across draws kept so the figure can show it.
        avg(col) = mean(skipmissing(per[!, col]))
        sdv(col) = ndraw > 1 ? std(skipmissing(per[!, col])) : missing
        push!(out, (f = f, weights = String(wmode), n_draws = ndraw,
                    n_variants = first(per.n_variants),
                    n_query_eff = avg(:n_query_eff), n_ref_eff = avg(:n_ref_eff),
                    n_shared = avg(:n_shared),
                    pr_query = avg(:pr_query), pr_ref = avg(:pr_ref),
                    effdim_query = avg(:effdim_query), effdim_ref = avg(:effdim_ref),
                    align_single = avg(:align_single),
                    align_double = avg(:align_double),
                    align_double_sd_draws = sdv(:align_double),
                    ceiling = avg(:ceiling), ceiling_sd = avg(:ceiling_sd),
                    ceiling_reps = P3_CEIL,
                    frac_ceiling = avg(:frac_ceiling),
                    frac_ceiling_sd_draws = sdv(:frac_ceiling),
                    lo = endpoint ? first(per.lo) : missing,
                    hi = endpoint ? first(per.hi) : missing,
                    frac_ceiling_lo = endpoint ? first(per.frac_ceiling_lo) : missing,
                    frac_ceiling_hi = endpoint ? first(per.frac_ceiling_hi) : missing,
                    nboot = endpoint ? P3_NBOOT : 0); promote = true)
    end
    mkpath(RESULTS)
    f = joinpath(RESULTS, wmode === :uniform ?
                          "part3_uk_vs_us_overlap_curve_unweighted.csv" :
                          "part3_uk_vs_us_overlap_curve.csv")
    CSV.write(f, out); @info "  → $f  ($(nrow(out)) rows)"
    out
end

function main_controls()
    which = isempty(ARGS) ? "all" : ARGS[1]
    which in ("task2", "task3", "all") ||
        error("usage: part3_overlap_controls.jl [task2|task3|all]")
    @info "nboot=$P3_NBOOT ceil=$P3_CEIL draws/f=$P3_DRAWS  results → $RESULTS"
    d = _load_p3()                      # load once, serve both tasks
    which in ("task2", "all") && task2(d)
    which in ("task3", "all") && task3(d)
    @info "Done."
end

abspath(PROGRAM_FILE) == (@__FILE__) && main_controls()
