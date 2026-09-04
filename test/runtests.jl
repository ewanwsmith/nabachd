using Test
using DataFrames
using Statistics
using CSV
using Random

include(joinpath(@__DIR__, "..", "src", "NABACHD.jl"))
using .NABACHD

Random.seed!(1234)   # determinism for the randomised fixtures below

@testset "NABACHD" begin

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Fisher transform" begin
        for r in (-0.99, -0.9, -0.3, 0.0, 0.42, 0.9, 0.99)
            @test fisher_zinv(fisher_z(r)) ≈ r atol = 1e-8
        end
        @test isfinite(fisher_z(1.0))       # clamped, no Inf
        @test isfinite(fisher_z(-1.0))
        @test fisher_z(0.0) == 0.0
        @test fisher_z(0.5) > 0             # monotone increasing
        @test fisher_z(0.5) < fisher_z(0.9)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "topk_mean" begin
        r = [0.9, 0.5, 0.1, -0.2]
        @test topk_mean(r, 1) ≈ 0.9
        @test topk_mean(r, 2) ≈ 0.7
        @test topk_mean(r, 4) ≈ mean(r)
        @test topk_mean(r, 10) ≈ mean(r)    # k clamped to length
        @test topk_mean(r, 0) ≈ 0.9         # k clamped up to 1
        @test topk_mean([-0.5, -0.9], 1) ≈ -0.5   # works with all-negative
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "weighted_neighbourhood_r" begin
        r = [0.9, 0.5, 0.1]
        @test weighted_neighbourhood_r(r, [1.0, 0.0, 0.0], 2) ≈ 0.9 atol = 1e-8
        got = weighted_neighbourhood_r(r, [1.0, 1.0, 1.0], 2)
        want = fisher_zinv(mean(fisher_z.([0.9, 0.5])))
        @test got ≈ want atol = 1e-8
        @test weighted_neighbourhood_r(r, [0.0, 0.0, 0.0], 2) ≈ want atol = 1e-8   # zero weights → uniform
        @test weighted_neighbourhood_r(r, [1.0, 1.0, 1.0], 2; fisher=false) ≈ 0.7 atol = 1e-8
        @test weighted_neighbourhood_r(r, [1.0, 1.0, 1.0], 10) ≈ fisher_zinv(mean(fisher_z.(r))) atol = 1e-8  # k clamp
        # selects by correlation, not position: unsorted input, weights follow the top-k
        r2 = [0.1, 0.95, 0.4]; w2 = [10.0, 1.0, 10.0]      # top-2 are idx 2 (0.95) and 3 (0.4)
        exp2 = fisher_zinv((1.0*fisher_z(0.95) + 10.0*fisher_z(0.4)) / 11.0)
        @test weighted_neighbourhood_r(r2, w2, 2) ≈ exp2 atol = 1e-8
        # a manual hand-weighted example
        exp3 = fisher_zinv((0.8*fisher_z(0.9) + 0.2*fisher_z(0.5)) / 1.0)
        @test weighted_neighbourhood_r(r, [0.8, 0.2, 0.0], 2) ≈ exp3 atol = 1e-8
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "correlation_matrix" begin
        Mq = randn(50, 2); Mr = randn(50, 3)
        R = correlation_matrix(Mq, Mr)
        @test size(R) == (2, 3)
        @test all(-1 .<= R .<= 1)
        # orientation: R[i,j] == cor(query i, reference j)
        for i in 1:2, j in 1:3
            @test R[i, j] ≈ cor(Mq[:, i], Mr[:, j]) atol = 1e-10
        end
        @test_throws Exception correlation_matrix(randn(10, 2), randn(9, 3))
        # perfect correlation and anti-correlation
        v = randn(30)
        @test correlation_matrix(reshape(v, :, 1), reshape(2v .+ 1, :, 1))[1, 1] ≈ 1.0 atol = 1e-10
        @test correlation_matrix(reshape(v, :, 1), reshape(-v, :, 1))[1, 1] ≈ -1.0 atol = 1e-10
        # zero-variance column → 0, not NaN
        Rz = correlation_matrix(reshape(v, :, 1), hcat(v, zeros(30)))
        @test !any(isnan, Rz)
        @test Rz[1, 2] == 0.0
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "default_k_sequence" begin
        ks = default_k_sequence(175)
        @test issorted(ks) && allunique(ks)
        @test first(ks) == 1 && last(ks) == 175
        @test issubset(1:50, ks)               # dense over the crossover region
        @test all(1 .<= ks .<= 175)
        ks2 = default_k_sequence(5)
        @test first(ks2) == 1 && last(ks2) == 5
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "calibrate_k / find_k_star" begin
        R = rand(20, 40)
        ks40 = default_k_sequence(40)
        cal = calibrate_k(R)
        @test names(cal) == ["k", "cor_peak", "cor_mean", "sd_topk"]
        @test nrow(cal) == length(ks40)
        @test cal.k == ks40
        @test issorted(cal.k)
        @test all(cal.sd_topk .>= 0)
        @test all(-1 .- 1e-9 .<= cal.cor_peak .<= 1 .+ 1e-9)
        @test all(-1 .- 1e-9 .<= cal.cor_mean .<= 1 .+ 1e-9)
        k, cal2 = find_k_star(R)
        @test 1 <= k <= 40
        @test cal2.k == cal.k
        # an explicit k_seq is honoured
        cal3 = calibrate_k(R; k_seq=[1, 5, 10])
        @test cal3.k == [1, 5, 10]
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "neighbourhood_scores" begin
        R = [0.9 0.5 0.1;
             0.2 0.8 0.7]
        freq = [1.0, 2.0, 1.0]
        qa = ["Q1", "Q2"]; ra = ["R1", "R2", "R3"]
        sc = neighbourhood_scores(R, qa, ra, freq; k=2)
        @test names(sc) == ["allele", "k", "peak_r", "mean_r", "nbhd_r",
                            "weighted_nbhd_r", "nearest_reference", "nearest_r"]
        @test issorted(sc.weighted_nbhd_r, rev=true)
        @test all(sc.peak_r .>= sc.mean_r)
        @test all(sc.peak_r .>= sc.nbhd_r)                 # top-k mean ≤ max
        # values consistent with the low-level functions, per query allele
        for a in qa
            i = findfirst(==(a), qa)
            row = sc[sc.allele .== a, :][1, :]
            @test row.peak_r ≈ maximum(R[i, :])
            @test row.mean_r ≈ mean(R[i, :])
            @test row.nbhd_r ≈ topk_mean(R[i, :], 2)
            @test row.weighted_nbhd_r ≈ weighted_neighbourhood_r(R[i, :], freq, 2)
            @test row.nearest_reference == ra[argmax(R[i, :])]
            @test row.nearest_r ≈ maximum(R[i, :])
        end
        @test all(sc.k .== 2)
        # dimension-mismatch guards
        @test_throws Exception neighbourhood_scores(R, ["only_one"], ra, freq; k=2)
        @test_throws Exception neighbourhood_scores(R, qa, ra, [1.0, 2.0]; k=2)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "load_wide" begin
        dir = mktempdir()
        wide = joinpath(dir, "wide.csv")
        CSV.write(wide, DataFrame(
            Gene=["A", "A", "B"], Locus=[1, 2, 1], Mutation=["m1", "m2", "m3"],
            Q_Q1=[1.0, 2.0, 3.0], Q_Q2=[0.0, 1.0, 2.0],
            R_R1=[3.0, 2.0, 1.0], R_R2=[1.0, 1.0, 1.0], other=[9, 9, 9]))
        q = load_wide(wide; key_cols=["Gene", "Locus", "Mutation"],
                      allele_prefix="Q_", strip_prefix="Q_")
        @test q.alleles == ["Q1", "Q2"]
        @test nvariants(q) == 3 && nalleles(q) == 2
        @test q.M[:, 1] == [1.0, 2.0, 3.0]
        @test eltype(q.keys.Locus) <: AbstractString          # keys stringified for stable joins
        # explicit columns instead of a prefix
        q2 = load_wide(wide; key_cols=["Gene", "Locus", "Mutation"],
                       allele_cols=["Q_Q1"], strip_prefix="Q_")
        @test q2.alleles == ["Q1"]
        # no strip → full column name kept
        q3 = load_wide(wide; key_cols=["Gene", "Locus", "Mutation"], allele_prefix="R_")
        @test Set(q3.alleles) == Set(["R_R1", "R_R2"])
        # errors
        @test_throws Exception load_wide(wide; key_cols=["Gene", "Locus", "Mutation"])
        @test_throws Exception load_wide(wide; key_cols=["Gene", "Locus", "Mutation"], allele_prefix="ZZ_")
        # missing cells → 0
        gappy = joinpath(dir, "gappy.csv")
        CSV.write(gappy, DataFrame(Frame=["HA"], Locus=[1], Mutation=["m"], A_x=[missing]))
        gp = load_wide(gappy; key_cols=["Frame", "Locus", "Mutation"], allele_prefix="A_", strip_prefix="A_")
        @test gp.M == reshape([0.0], 1, 1)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "load_long / load_cd8scape" begin
        dir = mktempdir()
        long = joinpath(dir, "long.csv")
        CSV.write(long, DataFrame(
            Gene=["A", "A", "A", "A", "B", "B"], Locus=[1, 1, 2, 2, 1, 1],
            Mutation=["m1", "m1", "m2", "m2", "m3", "m3"],
            MHC=["S1", "S2", "S1", "S2", "S1", "S2"],
            ELBR_A=[1.0, 5.0, 2.0, 4.0, 3.0, 3.0]))
        s = load_long(long; key_cols=["Gene", "Locus", "Mutation"],
                      allele_col="MHC", value_col="ELBR_A")
        @test Set(s.alleles) == Set(["S1", "S2"])
        @test nvariants(s) == 3

        # duplicate (variant, allele) rows are averaged
        dup = joinpath(dir, "dup.csv")
        CSV.write(dup, DataFrame(Frame=["HA", "HA"], Locus=[1, 1], Mutation=["m", "m"],
                                 MHC=["X", "X"], log2_foldchange_BR=[1.0, 3.0]))
        d = load_cd8scape(dup)
        @test nvariants(d) == 1 && d.alleles == ["X"]
        @test d.M[1, 1] ≈ 2.0

        # combine a LIST of files (different alleles, different variants; gaps → 0)
        longA = joinpath(dir, "A.csv"); longB = joinpath(dir, "B.csv")
        CSV.write(longA, DataFrame(Frame=["HA", "HA"], Locus=[1, 2], Mutation=["m1", "m2"],
                                   MHC=["X1", "X1"], log2_foldchange_BR=[0.5, 1.5]))
        CSV.write(longB, DataFrame(Frame=["HA", "MP"], Locus=[2, 1], Mutation=["m2", "m9"],
                                   MHC=["X2", "X2"], log2_foldchange_BR=[2.0, 3.0]))
        comb = load_cd8scape([longA, longB])
        @test Set(comb.alleles) == Set(["X1", "X2"])
        @test nvariants(comb) == 3                     # (HA,1,m1),(HA,2,m2),(MP,1,m9)
        ix1 = findfirst(==("X1"), comb.alleles); ix2 = findfirst(==("X2"), comb.alleles)
        # (MP,1,m9) has X2 but not X1 → X1 zero-filled there
        rowMP = findfirst(r -> r.Frame == "MP", eachrow(comb.keys))
        @test comb.M[rowMP, ix1] == 0.0
        @test comb.M[rowMP, ix2] == 3.0

        # value column really is selectable
        s_d = load_long(long; key_cols=["Gene", "Locus", "Mutation"], allele_col="MHC", value_col="ELBR_A")
        @test s_d.M[findfirst(r -> r.Locus == "1", eachrow(s_d.keys)),
                    findfirst(==("S2"), s_d.alleles)] ≈ 5.0
        # missing-column error
        @test_throws Exception load_long(long; key_cols=["Gene", "Locus", "Mutation"],
                                         allele_col="MHC", value_col="not_a_column")
        @test_throws Exception load_long(String[]; key_cols=["Frame"], allele_col="MHC", value_col="v")
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "frequencies + normalise" begin
        dir = mktempdir()
        freqf = joinpath(dir, "freq.csv")
        CSV.write(freqf, DataFrame(Allele=["HLA-A*0101", "HLA-B*0801"], Frequency=[0.23, 0.11]))
        fr = load_frequencies(freqf)
        @test haskey(fr, "HLA-A0101") && haskey(fr, "HLA-B0801")   # '*' stripped
        ps = ProfileSet(DataFrame(V=["1"]), ["HLA-A0101", "HLA-B0801", "HLA-C0999"], zeros(1, 3))
        @test frequency_vector(ps, fr) == [0.23, 0.11, 0.0]        # absent allele → 0
        @test frequency_vector(ps, nothing) == ones(3)            # unweighted
        # custom column names
        freqf2 = joinpath(dir, "freq2.csv")
        CSV.write(freqf2, DataFrame(name=["A"], f=[0.5]))
        fr2 = load_frequencies(freqf2; allele_col="name", freq_col="f")
        @test fr2["A"] == 0.5
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "read_allele_list + subset_alleles" begin
        dir = mktempdir()
        panelf = joinpath(dir, "panel.csv")
        CSV.write(panelf, DataFrame(Allele=["R1", "R1", "R3"], Locus=["a", "a", "b"]))
        @test Set(read_allele_list(panelf)) == Set(["R1", "R3"])   # unique
        ps = ProfileSet(DataFrame(V=["1", "2"]), ["R1", "R2", "R3"], reshape(collect(1.0:6.0), 2, 3))
        sub = subset_alleles(ps, ["R3", "R1"])
        @test sub.alleles == ["R1", "R3"]                          # order follows ps, not request
        @test nalleles(sub) == 2
        subw = @test_logs (:warn,) match_mode = :any subset_alleles(ps, ["R1", "NOPE"])
        @test subw.alleles == ["R1"]
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "align_profiles" begin
        keys = DataFrame(Frame=["HA", "HA", "MP"], Locus=["1", "2", "1"], Mutation=["a", "b", "c"])
        q = ProfileSet(keys, ["Q"], reshape([1.0, 2.0, 3.0], 3, 1))
        # same keys, shuffled order in reference → aligned back correctly
        keys_r = keys[[3, 1, 2], :]
        r = ProfileSet(keys_r, ["R"], reshape([30.0, 10.0, 20.0], 3, 1))
        Mq, Mr = align_profiles(q, r)
        @test size(Mq) == (3, 1) && size(Mr) == (3, 1)
        @test sort(vec(Mq)) == [1.0, 2.0, 3.0]
        # row order is join-dependent, but each row's query & reference correspond
        # to the SAME variant (here r == 10*q per variant), whatever the ordering
        @test vec(Mr) ≈ 10 .* vec(Mq)
        # partial overlap → only shared variants
        keys_p = DataFrame(Frame=["HA"], Locus=["2"], Mutation=["b"])
        rp = ProfileSet(keys_p, ["R"], reshape([99.0], 1, 1))
        Mq2, Mr2 = align_profiles(q, rp)
        @test size(Mq2, 1) == 1 && vec(Mq2) == [2.0] && vec(Mr2) == [99.0]
        # no shared variants → error
        keys_n = DataFrame(Frame=["ZZ"], Locus=["9"], Mutation=["z"])
        rn = ProfileSet(keys_n, ["R"], reshape([1.0], 1, 1))
        @test_throws Exception align_profiles(q, rn)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "analyse end-to-end" begin
        n = 300
        base1 = randn(n); base2 = randn(n)
        Mq = hcat(base1 .+ 0.01randn(n), base2 .+ 0.01randn(n))
        Mr = hcat(base1 .+ 0.5randn(n), randn(n), base2 .+ 0.5randn(n))
        keys = DataFrame(V=string.(1:n))
        q = ProfileSet(keys, ["Q1", "Q2"], Mq)
        r = ProfileSet(keys, ["R1", "R2", "R3"], Mr)

        res = analyse(q, r; k=2)
        @test res.k == 2
        @test size(res.R) == (2, 3)
        @test Set(res.scores.allele) == Set(["Q1", "Q2"])
        @test issorted(res.scores.weighted_nbhd_r, rev=true)
        @test all(res.scores.peak_r .>= res.scores.mean_r)
        @test res.scores[res.scores.allele .== "Q1", :].nearest_reference[1] == "R1"
        @test res.scores[res.scores.allele .== "Q2", :].nearest_reference[1] == "R3"

        # k=:auto returns an in-range integer plus a calibration table
        auto = analyse(q, r; k=:auto)
        @test 1 <= auto.k <= 3
        @test nrow(auto.calibration) >= 1

        # frequency weighting changes the weighted metric but not peak/mean/nbhd
        freqs = Dict("R1" => 0.9, "R2" => 0.05, "R3" => 0.05)
        wres = analyse(q, r; frequencies=freqs, k=2)
        @test wres.scores.peak_r == res.scores.peak_r || sort(wres.scores.peak_r) == sort(res.scores.peak_r)
        @test wres.frequencies == [0.9, 0.05, 0.05]
        # unweighted (nothing) == uniform weights
        u1 = analyse(q, r; frequencies=nothing, k=2)
        @test u1.frequencies == ones(3)

        # fixed k respected; fisher flag flows through
        @test analyse(q, r; k=1).k == 1
        nf = analyse(q, r; k=3, fisher=false)
        @test all(isfinite, nf.scores.weighted_nbhd_r)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "degenerate columns are robust (no NaN)" begin
        n = 120; b = randn(n)
        keys = DataFrame(V=string.(1:n))
        q = ProfileSet(keys, ["Q1"], reshape(b .+ 0.02randn(n), n, 1))
        r = ProfileSet(keys, ["R1", "Rdead"], hcat(b .+ 0.3randn(n), zeros(n)))
        res = @test_logs (:warn,) match_mode = :any analyse(q, r; k=1)
        @test !any(isnan, res.scores.weighted_nbhd_r)
        @test !any(isnan, res.scores.peak_r)
        @test res.scores.nearest_reference[1] == "R1"       # ignores the dead allele
        # constant query allele → all-zero row, no NaN, ranked with 0 similarity
        qc = ProfileSet(keys, ["Qconst"], reshape(fill(2.0, n), n, 1))
        rc = @test_logs (:warn,) match_mode = :any analyse(qc, r; k=1)
        @test rc.scores.peak_r[1] == 0.0
        @test !any(isnan, rc.scores.weighted_nbhd_r)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "show methods" begin
        ps = ProfileSet(DataFrame(V=["1", "2"]), ["A"], zeros(2, 1))
        @test occursin("ProfileSet", sprint(show, ps))
        q = ProfileSet(DataFrame(V=string.(1:20)), ["Q"], randn(20, 1))
        r = ProfileSet(DataFrame(V=string.(1:20)), ["R1", "R2"], randn(20, 2))
        nb = analyse(q, r; k=1)
        @test occursin("Neighbourhood", sprint(show, nb))
    end
end
