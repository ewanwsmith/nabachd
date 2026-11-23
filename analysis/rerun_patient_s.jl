#!/usr/bin/env julia
# ─────────────────────────────────────────────────────────────────────────────
# Re-run the Stanevich Patient S worked example under arcsin.
#
# Handover §7 item 4. The numbers in claude/genotype-representativeness-design-
# note.md §9 are from 2026-09-18, under Fisher-z, and every weighted_nbhd_r in
# them is stale. This reruns the same analysis with the same seed and nperm, so
# the null genotype DRAWS are identical and only the scoring changes — which
# isolates the transform and makes the two runs directly comparable.
#
# What must NOT move (if it does, something other than the transform changed):
#   k* = 6            — _k_star_reference averages raw correlations; it never
#                       sees the transform
#   n_variants        — 49,804
#   nearest_reference / nearest_r / shared  — raw correlations
#
# What will move: every weighted_nbhd_r, hence E, E_min, E_max, null_mean,
# null_sd, and therefore z, percentile and p.
#
# Writes to NEW filenames; the 18 Sep results are not touched.
#
#   julia --project=. analysis/rerun_patient_s.jl
#
# Override paths with NABACHD_GENO / NABACHD_STANEVICH if the folders moved.
# ─────────────────────────────────────────────────────────────────────────────

using Random, CSV, DataFrames

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

# The 18 Sep (Fisher-z) values, for the side-by-side at the end.
const FISHER = (E = 0.3986, E_min = 0.195, E_max = 0.661, z = -0.729,
                percentile = 0.250, p = 0.502,
                null_mean = 0.4383, null_sd = 0.0545, k = 6, n_variants = 49_804)

function need(path, what)
    isfile(path) || error("rerun_patient_s: missing $what\n  expected at: $path\n" *
                          "  set NABACHD_GENO / NABACHD_STANEVICH if the folders moved")
    path
end

function main()
    @info "Loading" REF_PROFILES PATIENT REF_FREQS
    need(REF_PROFILES, "reference escape profiles")
    need(PATIENT,      "patient escape profiles")
    need(REF_FREQS,    "reference carrier frequencies")

    reference = load_cd8scape(REF_PROFILES)
    patient   = load_cd8scape(PATIENT)
    # :error, not the default :warn — a duplicated allele row here would be a
    # per-haplotype frequency that no combining rule can turn into a carrier
    # frequency. Fix the file instead.
    freq      = load_frequencies(REF_FREQS; on_duplicate = :error)

    @info "Loaded" n_reference=nalleles(reference) n_patient=nalleles(patient) n_freq=length(freq)

    ec = escape_concordance(patient, reference;
                            frequencies = freq,
                            k           = :auto,
                            transform   = :arcsin,
                            nperm       = NPERM,
                            rng         = MersenneTwister(SEED))

    println("\n", ec, "\n")
    println("Per-allele escape concordance:")
    show(stdout, MIME("text/plain"), ec.per_allele)
    println("\n")

    mkpath(RESULTS)
    f_pa = joinpath(RESULTS, "patient_s_arcsin_per_allele.csv")
    f_su = joinpath(RESULTS, "patient_s_arcsin_summary.csv")
    CSV.write(f_pa, ec.per_allele)
    CSV.write(f_su, DataFrame(
        E = ec.E, E_min = ec.E_min, E_max = ec.E_max,
        z = ec.z, percentile = ec.percentile, p = ec.p,
        null_mean = ec.null_mean, null_sd = ec.null_sd, nperm = ec.nperm,
        k = ec.k, k_source = String(ec.k_source),
        n_genotype = ec.n_genotype, n_reference = ec.n_reference,
        n_variants = ec.n_variants, transform = "arcsin"))
    @info "  → $f_pa"
    @info "  → $f_su"

    # ── invariants ───────────────────────────────────────────────────────────
    println("Invariants (these must NOT have moved):")
    for (label, got, want) in (("k",          ec.k,          FISHER.k),
                               ("n_variants", ec.n_variants, FISHER.n_variants))
        println(rpad("  $label", 16), rpad(got, 10),
                got == want ? "OK (matches Fisher-z run)" :
                              "CHANGED — was $want; investigate before using this run")
    end

    # ── what moved ───────────────────────────────────────────────────────────
    println("\nFisher-z (18 Sep) → arcsin:")
    for (label, now, was) in (("E",          ec.E,          FISHER.E),
                              ("E_min",      ec.E_min,      FISHER.E_min),
                              ("E_max",      ec.E_max,      FISHER.E_max),
                              ("z",          ec.z,          FISHER.z),
                              ("percentile", ec.percentile, FISHER.percentile),
                              ("p",          ec.p,          FISHER.p),
                              ("null_mean",  ec.null_mean,  FISHER.null_mean),
                              ("null_sd",    ec.null_sd,    FISHER.null_sd))
        println(rpad("  $label", 14),
                rpad(round(was, digits=4), 10), "→ ", rpad(round(now, digits=4), 10),
                "(", round(now - was, digits=4), ")")
    end

    # ── per-allele null calibration ──────────────────────────────────────────
    println("\nPer-allele null — each allele against null alleles of its OWN locus:")
    pn = select(ec.per_allele, :allele, :weighted_nbhd_r, :null_locus,
                :n_null_allele, :null_mean_allele, :z_allele,
                :percentile_allele, :p_allele)
    show(stdout, MIME("text/plain"), sort(pn, :z_allele))
    println()
    println("This is what handover §7 item 5 asked for: E_min surfaces an")
    println("idiosyncratic allele, and z_allele/p_allele now say whether it is")
    println("idiosyncratic *for its locus* or merely typical of a locus whose")
    println("alleles all score low. Those are different claims.")

    println("\nQualitative check — the claim most at risk is that the B locus")
    println("drives low concordance and C07:02 scores highest. Order now:")
    for row in eachrow(sort(ec.per_allele, :weighted_nbhd_r, rev=true))
        println("  ", rpad(row.allele, 14), round(row.weighted_nbhd_r, digits=4))
    end
    return ec
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
