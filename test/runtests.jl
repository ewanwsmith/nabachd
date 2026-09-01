using Test
using DataFrames
using Statistics
using CSV

include(joinpath(@__DIR__, "..", "src", "NABACHD.jl"))
using .NABACHD

@testset "NABACHD" begin

    @testset "Fisher transform" begin
        for r in (-0.9, -0.3, 0.0, 0.42, 0.99)
            @test fisher_zinv(fisher_z(r)) ≈ r atol = 1e-8
        end
        @test isfinite(fisher_z(1.0))    # clamped, no Inf
        @test isfinite(fisher_z(-1.0))
    end

    @testset "topk_mean" begin
        r = [0.9, 0.5, 0.1, -0.2]
        @test topk_mean(r, 1) ≈ 0.9
        @test topk_mean(r, 2) ≈ 0.7
        @test topk_mean(r, 10) ≈ mean(r)   # k clamped to length
    end

    @testset "weighted_neighbourhood_r" begin
        r = [0.9, 0.5, 0.1]
        # a single dominant weight collapses to that allele's r
        @test weighted_neighbourhood_r(r, [1.0, 0.0, 0.0], 2) ≈ 0.9 atol = 1e-8
        # uniform weights == Fisher-z mean of the top-k
        got = weighted_neighbourhood_r(r, [1.0, 1.0, 1.0], 2)
        want = fisher_zinv(mean(fisher_z.([0.9, 0.5])))
        @test got ≈ want atol = 1e-8
        # all-zero weights fall back to uniform
        @test weighted_neighbourhood_r(r, [0.0, 0.0, 0.0], 2) ≈ want atol = 1e-8
        # non-Fisher path is a plain weighted mean
        @test weighted_neighbourhood_r(r, [1.0, 1.0, 1.0], 2; fisher=false) ≈ 0.7 atol = 1e-8
    end

    @testset "correlation_matrix" begin
        Mq = randn(50, 2); Mr = randn(50, 3)
        R = correlation_matrix(Mq, Mr)
        @test size(R) == (2, 3)
        @test all(-1 .<= R .<= 1)
        @test_throws Exception correlation_matrix(randn(10, 2), randn(9, 3))
    end

    @testset "k calibration / selection" begin
        R = rand(20, 40)
        cal = calibrate_k(R)
        @test names(cal) == ["k", "cor_peak", "cor_mean", "sd_topk"]
        @test issorted(cal.k)
        @test first(cal.k) == 1 && last(cal.k) == 40
        k, cal2 = find_k_star(R)
        @test 1 <= k <= 40
    end

    @testset "loaders (wide + long) and align" begin
        dir = mktempdir()

        wide = joinpath(dir, "wide.csv")
        CSV.write(wide, DataFrame(
            Gene=["A", "A", "B"], Locus=[1, 2, 1], Mutation=["m1", "m2", "m3"],
            Q_Q1=[1.0, 2.0, 3.0], Q_Q2=[0.0, 1.0, 2.0],
            R_R1=[3.0, 2.0, 1.0], R_R2=[1.0, 1.0, 1.0], other=[9, 9, 9]))
        q = load_wide(wide; key_cols=["Gene", "Locus", "Mutation"],
                      allele_prefix="Q_", strip_prefix="Q_")
        r = load_wide(wide; key_cols=["Gene", "Locus", "Mutation"],
                      allele_prefix="R_", strip_prefix="R_")
        @test q.alleles == ["Q1", "Q2"]
        @test nvariants(q) == 3 && nalleles(r) == 2
        @test q.M[:, 1] == [1.0, 2.0, 3.0]

        long = joinpath(dir, "long.csv")
        CSV.write(long, DataFrame(
            Gene=["A", "A", "A", "A", "B", "B"],
            Locus=[1, 1, 2, 2, 1, 1],
            Mutation=["m1", "m1", "m2", "m2", "m3", "m3"],
            MHC=["S1", "S2", "S1", "S2", "S1", "S2"],
            ELBR_A=[1.0, 5.0, 2.0, 4.0, 3.0, 3.0]))
        s = load_long(long; key_cols=["Gene", "Locus", "Mutation"],
                      allele_col="MHC", value_col="ELBR_A")
        @test Set(s.alleles) == Set(["S1", "S2"])
        @test nvariants(s) == 3

        # align a long query with the wide reference on shared variants
        Mq2, Mr2 = align_profiles(s, r)
        @test size(Mq2, 1) == size(Mr2, 1) == 3

        # combine a LIST of long files (different alleles, different variants)
        longA = joinpath(dir, "longA.csv"); longB = joinpath(dir, "longB.csv")
        CSV.write(longA, DataFrame(Frame=["HA","HA"], Locus=[1,2], Mutation=["m1","m2"],
                                   MHC=["X1","X1"], log2_foldchange_BR=[0.5,1.5]))
        CSV.write(longB, DataFrame(Frame=["HA","MP"], Locus=[2,1], Mutation=["m2","m9"],
                                   MHC=["X2","X2"], log2_foldchange_BR=[2.0,3.0]))
        combined = load_cd8scape([longA, longB])       # CD8scape defaults
        @test Set(combined.alleles) == Set(["X1","X2"])
        @test nvariants(combined) == 3                  # (HA,1,m1),(HA,2,m2),(MP,1,m9)
        # X1 present only in file A, X2 only in file B; gaps are zero-filled
        ix2 = findfirst(==("X2"), combined.alleles)
        @test any(==(3.0), combined.M[:, ix2])

        # missing entries become 0
        @test !any(ismissing, q.M)

        # frequencies + subsetting
        freqf = joinpath(dir, "freq.csv")
        CSV.write(freqf, DataFrame(Allele=["R1", "R2"], Frequency=[0.8, 0.2]))
        fr = load_frequencies(freqf)
        @test frequency_vector(r, fr) == [0.8, 0.2]
        @test frequency_vector(r, nothing) == [1.0, 1.0]
        rsub = subset_alleles(r, ["R1"])
        @test rsub.alleles == ["R1"]
    end

    @testset "normalise_allele (HLA star)" begin
        dir = mktempdir()
        freqf = joinpath(dir, "hla.csv")
        CSV.write(freqf, DataFrame(Allele=["HLA-A*0101"], Frequency=[0.23]))
        fr = load_frequencies(freqf)
        @test haskey(fr, "HLA-A0101")     # '*' stripped for matching
    end

    @testset "analyse end-to-end" begin
        # two query alleles: Q1 tracks R1 closely, Q2 tracks R3 closely
        n = 200
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
        # nearest reference makes sense
        row1 = res.scores[res.scores.allele .== "Q1", :][1, :]
        @test row1.nearest_reference == "R1"
    end
end
