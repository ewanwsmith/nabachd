#!/usr/bin/env julia
# ─────────────────────────────────────────────────────────────────────────────
# The cross-run gap: escape concordance under all four leave-one-out settings.
#
# Handover §7 item 3 — the open design problem. Asymmetric LOO is now one flag
# (§2b), so the remaining question is which combination gives the right
# estimand. This measures all four rather than arguing about them.
#
# The patient's E is computed CROSS-run (her CD8scape run vs the reference run),
# the null's WITHIN-run (reference vs reference). A shared allele therefore
# correlates with its own reference copy at ~0.91 for the patient but at exactly
# 1.0 within the reference. The four settings:
#
#   query=T null=T   default. Both self-matches dropped. Symmetric, and the
#                    question becomes "setting aside her own alleles' presence
#                    in the panel, do the OTHER local alleles read escape as
#                    she does?"
#   query=F null=T   the asymmetric option. Her cross-run self-match (~0.91) is
#                    kept as a real measurement; the null's exact 1.0 is dropped
#                    as an artefact of scoring a panel against itself. Question
#                    becomes "does the local population, INCLUDING her own
#                    alleles at their real replicate correlation, read escape as
#                    she does?"
#   query=T null=F   null inflated by its own exact 1.0s. Reported only to show
#                    the size of that inflation; not a candidate.
#   query=F null=F   both keep self. The null's 1.0s are artefacts, so this is
#                    not a candidate either.
#
# ⚠ KNOWN CONTAMINATION, read before interpreting query=F. HLA-B08:01's
# reference profile was APPENDED FROM THE PATIENT PANEL during the Path B
# reference build (design note §8 step 3) — it is literally her own column. Its
# cross-run "self" correlation is therefore exactly 1.0 by construction, where
# the genuinely independent shared alleles sit at 0.91–0.93. With query=F that
# 1.0 enters B08:01's neighbourhood as if it were a replicate measurement, and
# it is not. The script prints each shared allele's self-r so this is visible.
#
#   julia --project=. analysis/loo_sensitivity_patient_s.jl
#
# Same paths/env vars as analysis/rerun_patient_s.jl.
# ─────────────────────────────────────────────────────────────────────────────

using Random, CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "..", "src", "NABACHD.jl"))
using .NABACHD

const GENO = get(ENV, "NABACHD_GENO",
                 expanduser("~/Documents/PhD/nabachd_genotype_representativeness"))
const STAN = get(ENV, "NABACHD_STANEVICH",
                 expanduser("~/Documents/PhD/Stanevich_data/writeup_copy"))
const REF_PROFILES = joinpath(GENO, "data", "supertype", "per_allele_best_ranks_simulated.csv")
const REF_FREQS    = joinpath(GENO, "data", "supertype", "supertype_panel.csv")
const PATIENT      = joinpath(STAN, "panel", "per_allele_best_ranks_simulated.csv")
const RESULTS      = joinpath(GENO, "results")

const NPERM = parse(Int, get(ENV, "NABACHD_NPERM", "2000"))
const SEED  = parse(Int, get(ENV, "NABACHD_SEED",  "42"))

function need(path, what)
    isfile(path) || error("loo_sensitivity: missing $what\n  expected at: $path")
    path
end

function main()
    need(REF_PROFILES, "reference escape profiles")
    need(PATIENT,      "patient escape profiles")
    need(REF_FREQS,    "reference carrier frequencies")

    reference = load_cd8scape(REF_PROFILES)
    patient   = load_cd8scape(PATIENT)
    freq      = load_frequencies(REF_FREQS; on_duplicate = :error)
    @info "Loaded" n_reference=nalleles(reference) n_patient=nalleles(patient)

    # ── the cross-run self-correlations, which are what the gap is about ─────
    Mq, Mr = align_profiles(patient, reference)
    R = correlation_matrix(Mq, Mr)
    qn = NABACHD.normalise_allele.(patient.alleles)
    rn = NABACHD.normalise_allele.(reference.alleles)
    println("\nCross-run self-correlation of each shared allele")
    println("(the patient's column vs the reference's column for the SAME allele):")
    for (i, a) in enumerate(qn)
        j = findfirst(==(a), rn)
        if j === nothing
            println("  ", rpad(patient.alleles[i], 14), "not in reference panel")
        else
            r = R[i, j]
            flag = r > 0.999 ? "  ⚠ exactly 1.0 — this is the appended column, not a replicate" : ""
            println("  ", rpad(patient.alleles[i], 14), round(r, digits=4), flag)
        end
    end

    # ── all four LOO settings ────────────────────────────────────────────────
    rows = DataFrame()
    per  = DataFrame()
    for loo_q in (true, false), loo_n in (true, false)
        ec = escape_concordance(patient, reference;
                                frequencies = freq, k = :auto,
                                transform = :arcsin, nperm = NPERM,
                                leave_one_out_query = loo_q,
                                leave_one_out_null  = loo_n,
                                rng = MersenneTwister(SEED))
        push!(rows, (loo_query=loo_q, loo_null=loo_n,
                     E=ec.E, E_min=ec.E_min, E_max=ec.E_max,
                     z=ec.z, percentile=ec.percentile, p=ec.p,
                     null_mean=ec.null_mean, null_sd=ec.null_sd,
                     k=ec.k, nperm=ec.nperm); promote=true)
        pa = select(ec.per_allele, :allele, :weighted_nbhd_r)
        pa.loo_query .= loo_q ? "loo_TT_q" : "loo_FT_q"
        pa.loo_null  .= loo_n
        append!(per, pa; promote=true)
        @info "  done" loo_q loo_n E=round(ec.E, digits=4)
    end

    mkpath(RESULTS)
    f1 = joinpath(RESULTS, "patient_s_loo_sensitivity.csv")
    f2 = joinpath(RESULTS, "patient_s_loo_sensitivity_per_allele.csv")
    CSV.write(f1, rows); CSV.write(f2, per)
    @info "  → $f1"; @info "  → $f2"

    println("\nGenotype level, all four settings:")
    println(rpad("query/null", 14), rpad("E", 10), rpad("null_mean", 12),
            rpad("null_sd", 10), rpad("z", 10), rpad("pct", 9), "p")
    for r in eachrow(rows)
        println(rpad("$(r.loo_query ? "T" : "F")/$(r.loo_null ? "T" : "F")", 14),
                rpad(round(r.E, digits=4), 10),
                rpad(round(r.null_mean, digits=4), 12),
                rpad(round(r.null_sd, digits=4), 10),
                rpad(round(r.z, digits=3), 10),
                rpad(string(round(100*r.percentile, digits=1), "%"), 9),
                round(r.p, digits=4))
    end

    println("\nPer-allele, default (T/T) vs asymmetric (F/T):")
    d = unstack(filter(r -> r.loo_null, per), :allele, :loo_query, :weighted_nbhd_r)
    rename!(d, Dict("loo_TT_q" => "loo_TT", "loo_FT_q" => "loo_FT"))
    d.delta = d.loo_FT .- d.loo_TT
    show(stdout, MIME("text/plain"), sort(d, :delta, rev=true)); println()
    println("\nA large delta means that allele's score depends on keeping its own")
    println("cross-run self-match. For HLA-B08:01 that self-match is the appended")
    println("column (self-r = 1.0 above), so its delta is an artefact of the")
    println("reference build, not a property of the genotype.")
    return rows
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
