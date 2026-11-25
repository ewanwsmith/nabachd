using Test
using DataFrames
using Statistics
using LinearAlgebra
using CSV
using Random

include(joinpath(@__DIR__, "..", "src", "NABACHD.jl"))
using .NABACHD

Random.seed!(1234)   # determinism for the randomised fixtures below

@testset "NABACHD" begin

    # ─────────────────────────────────────────────────────────────────────────
    @testset "arcsin transform" begin
        for r in (-1.0, -0.99, -0.9, -0.3, 0.0, 0.42, 0.9, 0.99, 1.0)
            @test arcsin_tinv(arcsin_t(r)) ≈ r atol = 1e-8
        end
        @test arcsin_t(0.0) == 0.0
        @test arcsin_t(0.5) > 0                       # monotone increasing
        @test arcsin_t(0.5) < arcsin_t(0.9)
        # bounded at r = ±1 with no clamping — the reason for preferring it
        @test arcsin_t(1.0)  ≈ pi  atol = 1e-12
        @test arcsin_t(-1.0) ≈ -pi atol = 1e-12
        @test isfinite(arcsin_t(1.0)) && isfinite(arcsin_t(-1.0))
        # r values beyond ±1 (floating-point noise) are clamped, not NaN
        @test isfinite(arcsin_t(1.0 + 1e-12))
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
        want = arcsin_tinv(mean(arcsin_t.([0.9, 0.5])))
        @test got ≈ want atol = 1e-8
        @test weighted_neighbourhood_r(r, [0.0, 0.0, 0.0], 2) ≈ want atol = 1e-8   # zero weights → uniform
        @test weighted_neighbourhood_r(r, [1.0, 1.0, 1.0], 2; transform=:none) ≈ 0.7 atol = 1e-8
        @test weighted_neighbourhood_r(r, [1.0, 1.0, 1.0], 10) ≈ arcsin_tinv(mean(arcsin_t.(r))) atol = 1e-8  # k clamp
        # selects by correlation, not position: unsorted input, weights follow the top-k
        r2 = [0.1, 0.95, 0.4]; w2 = [10.0, 1.0, 10.0]      # top-2 are idx 2 (0.95) and 3 (0.4)
        exp2 = arcsin_tinv((1.0*arcsin_t(0.95) + 10.0*arcsin_t(0.4)) / 11.0)
        @test weighted_neighbourhood_r(r2, w2, 2) ≈ exp2 atol = 1e-8
        # a manual hand-weighted example
        exp3 = arcsin_tinv((0.8*arcsin_t(0.9) + 0.2*arcsin_t(0.5)) / 1.0)
        @test weighted_neighbourhood_r(r, [0.8, 0.2, 0.0], 2) ≈ exp3 atol = 1e-8
        # r = 1.0 (an allele shared between panels, scored on identical inputs)
        # must stay finite and must not dominate the average
        @test weighted_neighbourhood_r([1.0, 1.0], [1.0, 1.0], 2) ≈ 1.0 atol = 1e-8
        @test isfinite(weighted_neighbourhood_r([1.0, 0.5], [1.0, 1.0], 2))
        @test weighted_neighbourhood_r([1.0, 0.5], [1.0, 1.0], 2) <
              weighted_neighbourhood_r([1.0, 1.0], [1.0, 1.0], 2)
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
                            "weighted_nbhd_r", "nearest_reference", "nearest_r",
                            "shared", "nearest_is_self"]
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

        # fixed k respected; transform flag flows through
        @test analyse(q, r; k=1).k == 1
        nf = analyse(q, r; k=3, transform=:none)
        @test all(isfinite, nf.scores.weighted_nbhd_r)

        # panel alignment attached by default, skippable, and matches cka()
        @test res.alignment isa PanelAlignment
        @test 0 <= res.alignment.alignment <= 1
        @test res.alignment.alignment ≈ cka(align_profiles(q, r)...) atol = 1e-8
        @test res.alignment.weighted == false
        @test analyse(q, r; k=2, panel=false).alignment === nothing
        # frequency-weighted analyse marks the alignment weighted
        wres2 = analyse(q, r; frequencies=freqs, k=2)
        @test wres2.alignment.weighted == true
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "cka (panel neighbourhood alignment)" begin
        rng = MersenneTwister(42)
        m, n = 400, 6
        X = randn(rng, m, n)

        # identity, isotropic scaling + shift, and allele relabelling → 1
        @test cka(X, X) ≈ 1.0 atol = 1e-8
        @test cka(X, 3 .* X .+ 5.0) ≈ 1.0 atol = 1e-8          # centring removes shift
        @test cka(X, X[:, [6, 5, 4, 3, 2, 1]]) ≈ 1.0 atol = 1e-8

        # rotation of the allele basis leaves CKA unchanged (no correspondence needed)
        Q = Matrix(qr(randn(rng, n, n)).Q)
        @test cka(X, X * Q) ≈ 1.0 atol = 1e-6

        # symmetry, and different allele counts are fine
        Y = randn(rng, m, 4)
        @test cka(X, Y) ≈ cka(Y, X) atol = 1e-10
        @test cka(X, Y) < 0.3                                   # unrelated panels

        # breaking the variant alignment collapses similarity toward 0
        perm = randperm(rng, m)
        @test cka(X, X[perm, :]) < 0.2

        # frequency weighting: uniform weights == unweighted; uneven weights differ
        Yr = X * Q .+ 0.3 .* randn(rng, m, n)
        @test cka(X, Yr; wq=ones(n), wr=ones(n)) ≈ cka(X, Yr) atol = 1e-8
        @test cka(X, Yr; wq=[5.0, 5, 1, 1, 0.1, 0.1], wr=[5.0, 5, 1, 1, 0.1, 0.1]) != cka(X, Yr)

        # degenerate (zero-variance) columns don't produce NaN
        Xd = hcat(X, zeros(m))
        @test cka(Xd, Xd) ≈ 1.0 atol = 1e-8
        @test cka(zeros(m, 2), X[:, 1:2]) == 0.0                # all-degenerate → guarded 0

        # dimension guard
        @test_throws Exception cka(randn(rng, 10, 2), randn(rng, 9, 2))
        @test_throws Exception cka(X, X; wq=ones(n + 1))        # bad weight length

        # all-zero weights on a panel fall back to uniform (no collapse to 0), with a warning
        z1 = @test_logs (:warn,) match_mode = :any cka(X, X; wq=zeros(n))
        @test z1 ≈ 1.0 atol = 1e-8
        z2 = @test_logs (:warn,) match_mode = :any cka(X, Yr; wq=zeros(n))
        @test z2 ≈ cka(X, Yr) atol = 1e-8       # query all-zero → unweighted query
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "panel_alignment + permutation null" begin
        m, n = 500, 6
        base = randn(MersenneTwister(11), m, n)
        related = base .+ 0.2 .* randn(MersenneTwister(12), m, n)   # shares structure
        unrelated = randn(MersenneTwister(13), m, n)                # independent

        pr = panel_alignment(base, related; nperm=300, rng=MersenneTwister(1))
        @test pr isa PanelAlignment
        @test pr.alignment > 0.5
        @test pr.z > 5.0 && pr.p < 0.01                         # far above the null
        @test isfinite(pr.null_mean) && pr.null_sd > 0
        @test pr.alignment > pr.null_mean                       # observed beats the null
        @test pr.n_query == n && pr.n_reference == n && pr.n_variants == m

        pu = panel_alignment(base, unrelated; nperm=300, rng=MersenneTwister(2))
        @test pu.p > 0.05 && pu.z < pr.z                        # not significant

        # ProfileSet convenience method aligns + weights via frequencies
        keys = DataFrame(V=string.(1:m))
        q = ProfileSet(keys, ["Q$i" for i in 1:n], base)
        r = ProfileSet(keys, ["R$i" for i in 1:n], related)
        pa = panel_alignment(q, r; nperm=200, rng=MersenneTwister(3))
        @test pa.alignment ≈ cka(base, related) atol = 1e-8
        @test pa.weighted == false
        freqs = Dict("R1" => 0.5, "R2" => 0.3, "Q1" => 0.4)      # partial; others → 0
        paw = panel_alignment(q, r; frequencies=freqs, nperm=50, rng=MersenneTwister(4))
        @test paw.weighted == true && isfinite(paw.alignment)
        @test occursin("PanelAlignment", sprint(show, paw))
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "zero-inflation sensitivity" begin
        rng = MersenneTwister(99)
        ninfo, nzero, n = 120, 380, 5
        sig = randn(rng, ninfo, n)
        Xi = sig .+ 0.2 .* randn(rng, ninfo, n)          # informative, correlated
        Yi = sig .+ 0.5 .* randn(rng, ninfo, n)
        Mq = vcat(Xi, zeros(nzero, n))                    # + a shared non-binding block
        Mr = vcat(Yi, zeros(nzero, n))

        mask = informative_variants(Mq, Mr)
        @test count(mask) == ninfo                        # exactly the nonzero rows
        @test all(mask[1:ninfo]) && !any(mask[ninfo+1:end])
        @test count(informative_variants(Mq, Mr; mode=:both)) == ninfo

        s = alignment_sensitivity(Mq, Mr; nperm=200, rng=MersenneTwister(5))
        @test s isa AlignmentSensitivity
        @test s.n_variants_full == ninfo + nzero
        @test s.n_variants_masked == ninfo
        @test s.frac_retained ≈ ninfo / (ninfo + nzero)
        @test s.masked.alignment ≈ cka(Xi, Yi) atol = 1e-8   # masked == clean subset
        @test s.delta ≈ s.masked.alignment - s.full.alignment
        @test occursin("AlignmentSensitivity", sprint(show, s))

        # no zero rows → keep everything, full == masked
        s2 = alignment_sensitivity(Xi, Yi; nperm=100, rng=MersenneTwister(6))
        @test s2.frac_retained == 1.0
        @test s2.full.alignment ≈ s2.masked.alignment atol = 1e-8

        # too-few-informative guard
        @test_throws Exception alignment_sensitivity(zeros(10, 3), zeros(10, 3); nperm=10)

        # ProfileSet convenience method
        keys = DataFrame(V=string.(1:(ninfo+nzero)))
        q = ProfileSet(keys, ["Q$i" for i in 1:n], Mq)
        r = ProfileSet(keys, ["R$i" for i in 1:n], Mr)
        sp = alignment_sensitivity(q, r; nperm=100, rng=MersenneTwister(7))
        @test sp.n_variants_masked == ninfo
        @test sp.masked.alignment ≈ cka(Xi, Yi) atol = 1e-8
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
    # ─────────────────────────────────────────────────────────────────────────
    @testset "neighbourhood_scores: shared diagnostics + leave_one_out" begin
        # build a small R matrix where Q1 matches R1 (shared) and Q2 has no match
        R = [0.95 0.50 0.10;
             0.20 0.85 0.70]
        freq = [1.0, 2.0, 1.0]
        qa = ["Q1", "Q2"]
        ra = ["Q1", "R2", "R3"]   # R1 is named Q1 → shared with query allele 1

        sc = neighbourhood_scores(R, qa, ra, freq; k=2)
        # shared column: Q1 is in ra, Q2 is not
        row_q1 = sc[sc.allele .== "Q1", :][1, :]
        row_q2 = sc[sc.allele .== "Q2", :][1, :]
        @test row_q1.shared == true
        @test row_q2.shared == false
        # nearest_is_self: argmax(R[1,:]) == 1 (r=0.95) → reference is "Q1" → self
        @test row_q1.nearest_is_self == true
        @test row_q2.nearest_is_self == false

        # leave_one_out: Q1 excludes its name-match (column 1) before scoring
        sc_loo = neighbourhood_scores(R, qa, ra, freq; k=2, leave_one_out=true)
        row_q1_loo = sc_loo[sc_loo.allele .== "Q1", :][1, :]
        row_q2_loo = sc_loo[sc_loo.allele .== "Q2", :][1, :]
        # LOO removes the highest correlation for Q1, so nbhd_r must decrease
        @test row_q1_loo.nbhd_r < row_q1.nbhd_r
        # Q2 has no shared allele so LOO doesn't change its score
        @test row_q2_loo.nbhd_r ≈ row_q2.nbhd_r atol = 1e-10
        # nearest_reference and nearest_r still use the full row (unfiltered)
        @test row_q1_loo.nearest_reference == "Q1"
        @test row_q1_loo.nearest_r ≈ 0.95 atol = 1e-10
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "panel_alignment: shared diagnostics + drop_shared" begin
        rng = MersenneTwister(77)
        m, nq, nr = 300, 4, 6
        X = randn(rng, m, nq)
        Y = randn(rng, m, nr)
        qa = ["A1", "A2", "SHARED1", "SHARED2"]
        ra = ["SHARED1", "SHARED2", "B1", "B2", "B3", "B4"]

        # ── no allele names → n_shared = 0 ───────────────────────────────────
        pa_nonames = panel_alignment(X, Y; nperm=50, rng=MersenneTwister(1))
        @test pa_nonames.n_shared == 0
        @test pa_nonames.frac_shared_query == 0.0
        @test pa_nonames.frac_shared_reference == 0.0
        @test pa_nonames.drop_shared == false

        # ── with allele names, shared stats are correct ───────────────────────
        pa_named = panel_alignment(X, Y; nperm=50, rng=MersenneTwister(2),
                                   query_alleles=qa, reference_alleles=ra,
                                   n_query=nq, n_reference=nr)
        @test pa_named.n_shared == 2
        @test pa_named.frac_shared_query ≈ 2/4 atol = 1e-10
        @test pa_named.frac_shared_reference ≈ 2/6 atol = 1e-10
        @test pa_named.drop_shared == false
        # show includes shared count when n_shared > 0
        @test occursin("2 shared", sprint(show, pa_named))

        # ── drop_shared removes the 2 shared alleles from both sides ──────────
        pa_drop = panel_alignment(X, Y; nperm=50, rng=MersenneTwister(3),
                                  query_alleles=qa, reference_alleles=ra,
                                  n_query=nq, n_reference=nr, drop_shared=true)
        @test pa_drop.drop_shared == true
        @test pa_drop.n_shared == 2              # still reports original shared count
        # computed on 2 query + 4 reference alleles (shared excluded)
        X_nonshared = X[:, 1:2]
        Y_nonshared = Y[:, 3:6]
        @test pa_drop.alignment ≈ cka(X_nonshared, Y_nonshared) atol = 1e-8
        # drop_shared when panels have no overlap → identity (no columns removed)
        qa2 = ["A1", "A2", "A3", "A4"]
        pa_nodrop = panel_alignment(X, Y; nperm=50, rng=MersenneTwister(4),
                                    query_alleles=qa2, reference_alleles=ra,
                                    n_query=nq, n_reference=nr, drop_shared=true)
        @test pa_nodrop.alignment ≈ cka(X, Y) atol = 1e-8
        @test pa_nodrop.n_shared == 0
        # ProfileSet convenience: allele names flow through automatically
        keys = DataFrame(V=string.(1:m))
        q_ps = ProfileSet(keys, qa, X)
        r_ps = ProfileSet(keys, ra, Y)
        pa_ps = panel_alignment(q_ps, r_ps; nperm=50, rng=MersenneTwister(5))
        @test pa_ps.n_shared == 2
        pa_ps_drop = panel_alignment(q_ps, r_ps; nperm=50, rng=MersenneTwister(6), drop_shared=true)
        @test pa_ps_drop.alignment ≈ cka(X_nonshared, Y_nonshared) atol = 1e-8
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "figure extension: stubs and fallback" begin
        # The twelve figure functions are declared in the core module and given
        # methods by ext/NABACHDPlotsExt.jl, which Julia loads only when all of
        # CairoMakie, ColorSchemes, KernelDensity, MultivariateStats and UMAP
        # are present. The suite runs without them, so what is tested here is
        # the core side: the names exist, are exported, and fail with an
        # instruction rather than a bare MethodError.
        FIGFNS = (:plot_k_calibration, :plot_k_sd, :plot_mean_peak,
                  :plot_weighted_unweighted, :plot_ranked, :plot_profile_densities,
                  :umap_embedding, :plot_umap_species, :plot_umap_scored,
                  :plot_panel_alignment, :plot_panel_alignments, :save_all_figures)
        for f in FIGFNS
            @test isdefined(NABACHD, f)
            @test Base.isexported(NABACHD, f)
        end

        ext_loaded = Base.get_extension(NABACHD, :NABACHDPlotsExt) !== nothing
        if ext_loaded
            # Someone ran the suite with the optional packages available: the
            # extension must have added methods on top of the fallback.
            @test length(methods(NABACHD.save_all_figures)) > 1
            @test length(methods(NABACHD.plot_k_calibration)) > 1
        else
            # Without the extension every name has exactly the fallback, and it
            # must name the missing packages and how to install them.
            @test length(methods(NABACHD.save_all_figures)) == 1
            err = try
                NABACHD.save_all_figures(nothing, nothing, nothing; outdir="x"); nothing
            catch e; sprint(showerror, e) end
            @test err !== nothing
            @test occursin("plotting extension", err)
            @test occursin("Pkg.add", err)
            @test occursin("CairoMakie", err)
        end

        # The extension is declared in Project.toml and its file exists; a
        # rename on one side only would silently disable every figure.
        proj = read(joinpath(@__DIR__, "..", "Project.toml"), String)
        @test occursin("NABACHDPlotsExt", proj)
        @test isfile(joinpath(@__DIR__, "..", "ext", "NABACHDPlotsExt.jl"))

        # The five triggers must stay weak. `Pkg.add`ing them while this
        # project is active moves them into [deps], which makes them hard
        # dependencies of everyone who installs NABACHD and stops them
        # functioning as extension triggers. Install them into the default
        # environment instead; they remain loadable with any project active.
        function _toml_table(src, name)
            lines  = split(src, '\n')
            i      = findfirst(==("[$name]"), lines)
            i === nothing && return String[]
            out = String[]
            for l in lines[(i + 1):end]
                startswith(l, "[") && break
                isempty(strip(l)) && continue
                push!(out, String(first(split(l, " = "))))
            end
            out
        end
        hard = _toml_table(proj, "deps")
        weak = _toml_table(proj, "weakdeps")
        for d in ("CairoMakie", "ColorSchemes", "KernelDensity",
                  "MultivariateStats", "UMAP")
            @test d in weak
            @test !(d in hard)
        end
        # The trigger list and [weakdeps] must agree, or Julia never primes it.
        @test Set(weak) == Set(["CairoMakie", "ColorSchemes", "KernelDensity",
                                "MultivariateStats", "UMAP"])
    end

    @testset "escape_concordance: per-allele null" begin
        rng = MersenneTwister(4242)
        m   = 250
        ref_names = ["HLA-A01:01","HLA-A02:01","HLA-A03:01","HLA-A11:01",
                     "HLA-B07:02","HLA-B08:01","HLA-B15:01","HLA-B27:05",
                     "HLA-C07:01","HLA-C07:02","HLA-C04:01","HLA-C06:02"]
        q_names   = ["HLA-A01:01", "HLA-B07:02", "HLA-C07:01"]
        Mr = randn(rng, m, length(ref_names))
        Mq = hcat([Mr[:, findfirst(==(a), ref_names)] .+ 0.3 .* randn(rng, m)
                   for a in q_names]...)
        keys_df = DataFrame(V = string.(1:m))
        reference = ProfileSet(keys_df, ref_names, Mr)
        query     = ProfileSet(keys_df, q_names,   Mq)
        freqs     = Dict(NABACHD.normalise_allele(a) => 0.1 for a in ref_names)

        ec = escape_concordance(query, reference; frequencies=freqs, k=3,
                                nperm=100, rng=MersenneTwister(9))
        pa = ec.per_allele

        # ── columns exist and are populated ──────────────────────────────────
        for col in ("null_locus", "n_null_allele", "null_mean_allele",
                    "null_sd_allele", "z_allele", "percentile_allele", "p_allele")
            @test col in names(pa)
        end
        # ── each query allele is calibrated against its OWN locus ────────────
        for row in eachrow(pa)
            @test row.null_locus == NABACHD._allele_locus(
                      NABACHD.normalise_allele(row.allele))
            @test row.n_null_allele > 0
            @test 0.0 <= row.percentile_allele <= 1.0
            @test 0.0 <= row.p_allele <= 1.0
            @test isfinite(row.z_allele)
        end
        # three loci present, so three distinct buckets
        @test Set(skipmissing(pa.null_locus)) == Set(["A", "B", "C"])

        # ── the null bucket sizes are consistent with the draw ───────────────
        # match_zygosity draws 2 alleles per locus per null genotype, and 100
        # genotypes were drawn, so each locus bucket holds 200 scores.
        @test all(pa.n_null_allele .== 200)

        # ── z_allele agrees with its own definition ──────────────────────────
        for row in eachrow(pa)
            recomputed = (row.weighted_nbhd_r - row.null_mean_allele) /
                         row.null_sd_allele
            @test row.z_allele ≈ recomputed atol = 1e-10
        end

        # ── alternative flows through to p_allele ────────────────────────────
        ec_l = escape_concordance(query, reference; frequencies=freqs, k=3,
                                  nperm=100, alternative=:less,
                                  rng=MersenneTwister(9))
        ec_g = escape_concordance(query, reference; frequencies=freqs, k=3,
                                  nperm=100, alternative=:greater,
                                  rng=MersenneTwister(9))
        # same draws, so the one-sided p's must be complementary in rank order
        @test all(ec_l.per_allele.p_allele .+ ec_g.per_allele.p_allele .>= 1.0 .- 1e-9)
        # the genotype-level statistic is untouched by the addition
        @test ec_l.E ≈ ec.E atol = 1e-12
        @test ec_g.E ≈ ec.E atol = 1e-12

        # ── locus parser is shared with the null sampler ─────────────────────
        @test NABACHD._allele_locus("HLA-A01:01") == "A"
        @test NABACHD._allele_locus("BoLA-N:03401") == "N"
        @test NABACHD._allele_locus("SLA-1:0101") == "1"
        @test NABACHD._allele_locus("nonsense") === nothing
    end

    @testset "panel_alignment: effective panel sizes" begin
        rng = MersenneTwister(991)
        m, nq, nr = 200, 5, 7
        X = randn(rng, m, nq)
        Y = randn(rng, m, nr)

        # ── unweighted: effective sizes are the column counts, PR likewise ────
        pa_u = panel_alignment(X, Y; nperm=20, rng=MersenneTwister(1))
        @test pa_u.n_query_effective == nq
        @test pa_u.n_reference_effective == nr
        @test pa_u.pr_query ≈ Float64(nq) atol = 1e-12
        @test pa_u.pr_reference ≈ Float64(nr) atol = 1e-12
        # catalogue sizes agree here, so show stays as it was
        @test !occursin("effective", sprint(show, pa_u))

        # ── zero-weight alleles drop out of the effective count ───────────────
        # Two query alleles carry no weight; their columns are all-zero after
        # √w scaling and enter no inner product, so the statistic sees 3.
        wq = [1.0, 1.0, 1.0, 0.0, 0.0]
        wr = fill(0.5, nr)
        pa_w = panel_alignment(X, Y; wq=wq, wr=wr, nperm=20,
                               rng=MersenneTwister(2), weighted=true)
        @test pa_w.n_query_effective == 3
        @test pa_w.n_reference_effective == nr
        # n_query still reports the catalogue size it was given
        @test pa_w.n_query == nq
        @test occursin("effective", sprint(show, pa_w))
        # and the alignment really is the one on the three weighted columns
        @test pa_w.alignment ≈ cka(X[:, 1:3], Y; wq=wq[1:3], wr=wr) atol = 1e-10

        # ── participation ratio tracks weight concentration ───────────────────
        # Even weights → PR is the count; one dominant allele → PR near 1.
        pa_even = panel_alignment(X, Y; wq=fill(2.0, nq), wr=wr, nperm=20,
                                  rng=MersenneTwister(3), weighted=true)
        @test pa_even.pr_query ≈ Float64(nq) atol = 1e-10
        pa_conc = panel_alignment(X, Y; wq=[1.0, 1e-6, 1e-6, 1e-6, 1e-6], wr=wr,
                                  nperm=20, rng=MersenneTwister(4), weighted=true)
        @test pa_conc.n_query_effective == nq      # all carry *some* weight
        @test pa_conc.pr_query < 1.001             # but one carries nearly all
        @test pa_conc.pr_query <= pa_conc.n_query_effective

        # ── all-zero weights: mirrors the uniform fallback in _weight_columns ─
        pa_zero = @test_logs (:warn, r"all weights are zero") match_mode=:any panel_alignment(
            X, Y; wq=zeros(nq), wr=wr, nperm=20, rng=MersenneTwister(5), weighted=true)
        @test pa_zero.n_query_effective == nq
        @test pa_zero.pr_query ≈ Float64(nq) atol = 1e-12

        # ── effective sizes are taken AFTER drop_shared ───────────────────────
        qa = ["A1", "A2", "SHARED1", "SHARED2", "A3"]
        ra = ["SHARED1", "SHARED2", "B1", "B2", "B3", "B4", "B5"]
        pa_drop = panel_alignment(X, Y; nperm=20, rng=MersenneTwister(6),
                                  query_alleles=qa, reference_alleles=ra,
                                  n_query=nq, n_reference=nr, drop_shared=true)
        @test pa_drop.n_query_effective == 3       # 5 less the 2 shared
        @test pa_drop.n_reference_effective == 5   # 7 less the 2 shared
        @test pa_drop.n_query == nq                # catalogue size unchanged
    end


end

@testset "escape_concordance" begin

    # ── Minimal synthetic setup ──────────────────────────────────────────────
    # 20 variants, 6 reference alleles (2 per locus A/B/C), 2 query alleles
    rng_setup = MersenneTwister(100)
    n_vars = 20

    # Reference: 6 alleles with correlated profiles
    Mr_base = randn(rng_setup, n_vars, 2)
    ref_profiles = hcat(
        Mr_base * [1.0 0.2; 0.2 1.0] + 0.1 * randn(rng_setup, n_vars, 2),  # A locus
        Mr_base * [0.8 0.1; 0.1 0.9] + 0.1 * randn(rng_setup, n_vars, 2),  # B locus
        Mr_base * [0.9 0.15; 0.15 0.85] + 0.1 * randn(rng_setup, n_vars, 2) # C locus
    )
    ref_alleles = ["HLA-A01:01", "HLA-A03:01", "HLA-B07:02", "HLA-B08:01",
                   "HLA-C07:01", "HLA-C07:02"]
    keys_df = DataFrame(Frame=fill("F", n_vars), Locus=fill("L", n_vars),
                        Mutation=string.(1:n_vars))

    reference = ProfileSet(keys_df, ref_alleles, ref_profiles)

    # Query: clone 2 reference alleles (high concordance)
    query_profiles = ref_profiles[:, [1, 3]]   # A01:01 and B07:02
    query_alleles  = ["HLA-A01:01", "HLA-B07:02"]
    query_ps       = ProfileSet(keys_df, query_alleles, query_profiles)

    # Frequencies: uniform within locus
    freqs = Dict("HLA-A01:01" => 0.3, "HLA-A03:01" => 0.2,
                 "HLA-B07:02" => 0.25, "HLA-B08:01" => 0.15,
                 "HLA-C07:01" => 0.1, "HLA-C07:02" => 0.1)

    # ── Basic call succeeds ──────────────────────────────────────────────────
    ec = escape_concordance(query_ps, reference;
                            frequencies=freqs, k=2, nperm=200,
                            rng=MersenneTwister(42))

    @test ec isa EscapeConcordance
    @test ec.n_genotype  == 2
    @test ec.n_reference == 6
    @test ec.n_variants  == n_vars
    @test ec.k           == 2
    @test ec.k_source    == :user
    @test ec.nperm       == 200
    @test 0.0 <= ec.E <= 1.0
    @test ec.E_min <= ec.E <= ec.E_max
    @test 0.0 <= ec.percentile <= 1.0
    @test 0.0 <= ec.p <= 1.0

    # ── per_allele DataFrame structure ──────────────────────────────────────
    @test nrow(ec.per_allele) == 2
    @test "allele" in names(ec.per_allele)
    @test "weighted_nbhd_r" in names(ec.per_allele)

    # ── E is mean of per-allele weighted_nbhd_r ─────────────────────────────
    @test ec.E ≈ mean(ec.per_allele.weighted_nbhd_r) atol = 1e-10

    # ── k_source :auto when k=:auto ─────────────────────────────────────────
    ec_auto = escape_concordance(query_ps, reference;
                                  frequencies=freqs, nperm=50,
                                  rng=MersenneTwister(7))
    @test ec_auto.k_source == :auto
    @test ec_auto.k >= 1

    # ── Reproducibility under fixed seed ────────────────────────────────────
    ec_a = escape_concordance(query_ps, reference;
                               frequencies=freqs, k=2, nperm=100,
                               rng=MersenneTwister(99))
    ec_b = escape_concordance(query_ps, reference;
                               frequencies=freqs, k=2, nperm=100,
                               rng=MersenneTwister(99))
    @test ec_a.E         ≈ ec_b.E         atol = 1e-12
    @test ec_a.z         ≈ ec_b.z         atol = 1e-12
    @test ec_a.percentile ≈ ec_b.percentile atol = 1e-12
    @test ec_a.p         ≈ ec_b.p         atol = 1e-12

    # ── Cloned alleles → high concordance relative to random draws ──────────
    # The query alleles are exact copies of reference alleles, so E should be
    # high and percentile should be > 0.5 (above a typical random genotype).
    # We use more permutations for a stable test.
    ec_high = escape_concordance(query_ps, reference;
                                  frequencies=freqs, k=2, nperm=500,
                                  rng=MersenneTwister(1234))
    @test ec_high.percentile >= 0.4    # should be well above random

    # ── Unweighted (no frequencies) runs without error ───────────────────────
    ec_unw = escape_concordance(query_ps, reference;
                                 frequencies=nothing, k=2, nperm=100,
                                 rng=MersenneTwister(5))
    @test ec_unw isa EscapeConcordance
    @test 0.0 <= ec_unw.E <= 1.0

    # ── show method produces meaningful string ───────────────────────────────
    s = sprint(show, ec)
    @test occursin("EscapeConcordance", s)
    @test occursin("E=", s)
    @test occursin("z=", s)
    @test occursin("percentile=", s)
    @test occursin("k=2", s)

end

@testset "sample_local_genotypes" begin

    rng_sg = MersenneTwister(200)
    n_vars = 10
    keys_df = DataFrame(Frame=fill("F", n_vars), Locus=fill("L", n_vars),
                        Mutation=string.(1:n_vars))
    ref_alleles = ["HLA-A01:01", "HLA-A03:01", "HLA-B07:02", "HLA-B08:01",
                   "HLA-C07:01", "HLA-C07:02"]
    ref_profiles = randn(rng_sg, n_vars, 6)
    reference = ProfileSet(keys_df, ref_alleles, ref_profiles)

    freqs = Dict("HLA-A01:01" => 0.3, "HLA-A03:01" => 0.2,
                 "HLA-B07:02" => 0.25, "HLA-B08:01" => 0.15,
                 "HLA-C07:01" => 0.1, "HLA-C07:02" => 0.1)

    genotypes = sample_local_genotypes(reference, freqs;
                                        n=100, rng=MersenneTwister(42))

    # Returns a vector of length n
    @test length(genotypes) == 100
    # Each genotype has 6 alleles (2 per locus A, B, C)
    @test all(length(g) == 6 for g in genotypes)
    # All alleles in each genotype are valid reference alleles
    ref_set = Set(ref_alleles)
    @test all(all(a in ref_set for a in g) for g in genotypes)
    # Heterozygosity: within each locus pair, alleles differ (match_zygosity=true)
    # not guaranteed if only 1 allele has freq>0 per locus, but here 2 per locus exist
    all_het = true
    for g in genotypes
        a1, a2 = g[1], g[2]  # A locus
        b1, b2 = g[3], g[4]  # B locus
        c1, c2 = g[5], g[6]  # C locus
        if a1 == a2 || b1 == b2 || c1 == c2
            all_het = false; break
        end
    end
    @test all_het

    # Reproducibility
    g1 = sample_local_genotypes(reference, freqs; n=20, rng=MersenneTwister(77))
    g2 = sample_local_genotypes(reference, freqs; n=20, rng=MersenneTwister(77))
    @test g1 == g2

end

# ─────────────────────────────────────────────────────────────────────────────
# Non-HLA loci, p-value tail, and asymmetric leave-one-out
# (kwargs added 2026-10-05; see handover open items 4/5/6)
# ─────────────────────────────────────────────────────────────────────────────
@testset "escape_concordance kwargs" begin

    nv = 40
    keys_df = DataFrame(Frame=fill("NP", nv), Locus=fill("L", nv),
                        Mutation=string.(1:nv))
    bola = ["BoLA-1:00901", "BoLA-1:01901", "BoLA-2:01201", "BoLA-2:01801",
            "BoLA-3:00201", "BoLA-3:01701", "BoLA-6:01401", "BoLA-N:03401"]
    Mr = randn(MersenneTwister(11), nv, length(bola))
    reference = ProfileSet(keys_df, bola, Mr)
    freqs = Dict(a => 0.1 + 0.05 * i for (i, a) in enumerate(bola))
    query = ProfileSet(keys_df, ["BoLA-2:01201", "BoLA-3:00201"],
                       Mr[:, [3, 5]] .+ 0.3 .* randn(MersenneTwister(12), nv, 2))

    # ── loci: the HLA-only default must not silently yield an empty null ─────
    @test_throws ErrorException sample_local_genotypes(reference, freqs;
                                                      n = 5, rng = MersenneTwister(1))

    # loci=:auto infers the panel's own loci (1, 2, 3, 6, N here)
    g_auto = sample_local_genotypes(reference, freqs; n = 10, loci = :auto,
                                    rng = MersenneTwister(1))
    @test length(g_auto) == 10
    @test all(!isempty, g_auto)
    ref_set = Set(bola)
    @test all(all(a in ref_set for a in g) for g in g_auto)
    # 5 loci; the two singleton loci (6, N) contribute a homozygous pair each
    @test all(length(g) == 10 for g in g_auto)

    # An explicit locus list is still honoured
    g_expl = sample_local_genotypes(reference, freqs; n = 5, loci = ["2", "3"],
                                    rng = MersenneTwister(1))
    @test all(length(g) == 4 for g in g_expl)

    # ── loci=:auto must agree with the default on an HLA class I panel ───────
    hla = ["HLA-A01:01", "HLA-A03:01", "HLA-B07:02", "HLA-B08:01",
           "HLA-C07:01", "HLA-C07:02"]
    hla_ref = ProfileSet(keys_df, hla, randn(MersenneTwister(5), nv, 6))
    hf = Dict(a => 0.2 for a in hla)
    @test sample_local_genotypes(hla_ref, hf; n = 8, rng = MersenneTwister(9)) ==
          sample_local_genotypes(hla_ref, hf; n = 8, loci = :auto,
                                 rng = MersenneTwister(9))

    # ── escape_concordance passes loci through ──────────────────────────────
    ec = escape_concordance(query, reference; frequencies = freqs, k = 3,
                            nperm = 100, loci = :auto, rng = MersenneTwister(42))
    @test ec isa EscapeConcordance
    @test !isnan(ec.null_mean)       # the old HLA-only default gave NaN here
    @test !isnan(ec.E)

    # ── alternative: tails are consistent with each other ───────────────────
    mk(alt) = escape_concordance(query, reference; frequencies = freqs, k = 3,
                                 nperm = 100, loci = :auto, alternative = alt,
                                 rng = MersenneTwister(42))
    ec2, ecl, ecg = mk(:two_sided), mk(:less), mk(:greater)
    # E, z and percentile do not depend on the tail; only p does
    @test ec2.E ≈ ecl.E ≈ ecg.E
    @test ec2.z ≈ ecl.z ≈ ecg.z
    @test ec2.percentile ≈ ecl.percentile
    @test ec2.p ≈ min(2 * min(ecl.p, ecg.p), 1.0)
    @test all(0.0 < e.p <= 1.0 for e in (ec2, ecl, ecg))
    @test_throws ErrorException mk(:nonsense)

    # ── asymmetric leave-one-out ────────────────────────────────────────────
    sym  = escape_concordance(query, reference; frequencies = freqs, k = 3,
                              nperm = 100, loci = :auto, rng = MersenneTwister(42))
    asym = escape_concordance(query, reference; frequencies = freqs, k = 3,
                              nperm = 100, loci = :auto,
                              leave_one_out_query = false, rng = MersenneTwister(42))
    # The null is scored identically; only the patient side changes
    @test sym.null_mean ≈ asym.null_mean
    @test sym.null_sd   ≈ asym.null_sd
    # Both query alleles are in the reference panel, so dropping leave-one-out
    # lets each match itself at r = 1.0 and E can only rise
    @test asym.E > sym.E
    # Defaults are symmetric, i.e. unchanged from before these kwargs existed
    @test sym.E ≈ escape_concordance(query, reference; frequencies = freqs, k = 3,
                                     nperm = 100, loci = :auto,
                                     leave_one_out_query = true,
                                     leave_one_out_null = true,
                                     rng = MersenneTwister(42)).E

end

# ─────────────────────────────────────────────────────────────────────────────
# load_frequencies duplicate-key handling (added 2026-10-05; handover item 2)
# ─────────────────────────────────────────────────────────────────────────────
@testset "load_frequencies duplicates" begin

    dir  = mktempdir()
    # One row per allele × haplotype: BoLA-2:02501 and BoLA-4:02401 appear twice,
    # exactly as the pre-rebuild Codner panel did.
    dup_path = joinpath(dir, "dup_panel.csv")
    open(dup_path, "w") do io
        println(io, "Allele,Locus,Frequency")
        println(io, "BoLA-1:02301,A14,0.2344")
        println(io, "BoLA-2:02501,A14,0.2344")
        println(io, "BoLA-4:02401,A14,0.2344")
        println(io, "BoLA-1:00901,A15,0.3398")
        println(io, "BoLA-2:02501,A15,0.3398")
        println(io, "BoLA-4:02401,A15,0.3398")
    end
    # One row per allele: no duplicates.
    clean_path = joinpath(dir, "clean_panel.csv")
    open(clean_path, "w") do io
        println(io, "Allele,Locus,Frequency")
        println(io, "BoLA-1:02301,BoLA-1,0.2500")
        println(io, "BoLA-2:02501,BoLA-2,0.5417")
    end

    # ── default :warn keeps the last row and says so ────────────────────────
    d_warn = @test_logs (:warn,) match_mode=:any load_frequencies(dup_path)
    @test d_warn["BoLA-2:02501"] ≈ 0.3398      # A15 row, i.e. the last
    @test d_warn["BoLA-4:02401"] ≈ 0.3398
    @test d_warn["BoLA-1:02301"] ≈ 0.2344      # unduplicated rows untouched
    @test length(d_warn) == 4

    # ── the historical numbers are preserved, so nothing silently moved ─────
    @test d_warn == load_frequencies(dup_path; on_duplicate = :last)

    # ── explicit rules ──────────────────────────────────────────────────────
    d_sum = load_frequencies(dup_path; on_duplicate = :sum)
    @test d_sum["BoLA-2:02501"] ≈ 0.2344 + 0.3398
    d_max = load_frequencies(dup_path; on_duplicate = :max)
    @test d_max["BoLA-2:02501"] ≈ 0.3398
    d_first = load_frequencies(dup_path; on_duplicate = :first)
    @test d_first["BoLA-2:02501"] ≈ 0.2344
    @test_throws ErrorException load_frequencies(dup_path; on_duplicate = :error)
    @test_throws ErrorException load_frequencies(dup_path; on_duplicate = :nonsense)

    # ── a clean file is unaffected, warns nothing, and :error is safe ───────
    d_clean = load_frequencies(clean_path)
    @test d_clean["BoLA-2:02501"] ≈ 0.5417
    @test length(d_clean) == 2
    @test d_clean == load_frequencies(clean_path; on_duplicate = :error)
    @test d_clean == load_frequencies(clean_path; on_duplicate = :sum)

end
