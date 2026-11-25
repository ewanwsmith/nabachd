# NABACHD

**N**eighbourhood **A**nalysis of **B**inding-escape **A**gainst **C**lass-I
**H**LA **D**ata

*nàbachd* /ˈnaːbəxk/ — neighbourhood in Scottish Gaelic

A Julia implementation of a metric of **MHC-I neighbourhoods**: a framework for
quantifying the *meaningfulness* of MHC-I binding escape from one allele panel
relative to another, from
[CD8scape](https://doi.org/10.64898/2026.04.20.719634) output.

The package implements two methods. **Per allele:** for each allele in a **query**
panel (e.g. cattle BoLA), its per-variant escape profile is correlated against
every allele in a larger, optionally frequency-weighted **reference** panel
(e.g. the human HLA repertoire); a neighbourhood size *k* is chosen
automatically, and the **frequency-weighted top-*k* neighbourhood r** summarises
how closely that query allele's escape landscape is mirrored by the reference.
**Per panel:** a single **neighbourhood alignment** — the frequency-weighted
centred kernel alignment (CKA) between the two panels, double-centred and
reported against an attainable **ceiling** — scores whether the two panels
organise the *variant* escape landscape the same way. Unlike the per-allele metric, the alignment is **symmetric**:
`alignment(A, B) = alignment(B, A)`, so the **query** / **reference** labels
(which do matter for the per-allele step) are irrelevant to it — neither panel
is privileged. It is bounded in [0, 1] and comparable across panel pairs of
different sizes. Both panels are fully configurable, so the method applies to any
pair of panels netMHCpan / CD8scape can handle — a non-human panel against the
human HLA repertoire, or one human panel against another (e.g. across
populations), not only BoLA vs HLA.

It reads **raw CD8scape output directly** and, in the style of CD8scape, is
driven by a command you point at your data. You pass a list of
`per_allele_best_ranks.csv` files (or their run folders) and it combines them
internally — no pre-processing.

## Installation

Requires Julia ≥ 1.10 ([juliaup](https://github.com/JuliaLang/juliaup)
recommended).

```julia
using Pkg
Pkg.add(url = "https://github.com/ewanwsmith/nabachd")
```

or, to work on it:

```bash
git clone https://github.com/ewanwsmith/nabachd.git
cd nabachd
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The core module depends only on `CSV`, `DataFrames`, `LinearAlgebra`, `Random`
and `Statistics`, so a normal install is light. Figures live in a **package
extension** (`ext/NABACHDPlotsExt.jl`) whose dependencies — `CairoMakie`,
`UMAP`, `MultivariateStats`, `KernelDensity`, `ColorSchemes` — are declared as
weak and are **not** installed with the package. Add them and the extension
loads itself:

```julia
using Pkg; Pkg.add(["CairoMakie","ColorSchemes","KernelDensity",
                    "MultivariateStats","UMAP"])
using NABACHD, CairoMakie      # the extension loads here
```

## Method

Given CD8scape escape profiles for a query and a reference panel across a shared
set of variants:

1. **Correlation matrix.** Pearson *r* between every query and reference profile
   → an `n_query × n_reference` matrix `R`. Missing / non-binding scores are 0.
2. **Reference summaries** per query allele *i*: `peak_r = maxⱼ rᵢⱼ` (depth of
   the closest single match) and `mean_r = meanⱼ rᵢⱼ` (breadth across the
   repertoire).
3. **Neighbourhood size k\*** — the crossover at which the unweighted top-*k*
   score is equally predictive of both extremes:

   ```
   k* = argzero_k [ cor(top-k, peak_r) − cor(top-k, mean_r) ]
   ```

   evaluated across all query alleles; `k*` is the last integer at which the
   score is still peak-dominant (⌊crossover⌋). Override with `--k`.
4. **Weighted neighbourhood r** (primary *per-allele* metric): for the *k*
   nearest reference alleles, weight their correlations by reference (carrier)
   frequency, renormalise within the neighbourhood, and average on the Fisher
   *z* scale (back-transformed). With no frequency file the analysis is
   unweighted.

5. **Neighbourhood alignment** (the *panel-level* metric): a single score
   **between** the two panels — the query/reference roles from the per-allele
   step do not apply here, the alignment is symmetric and treats the panels
   interchangeably. Where `weighted_nbhd_r` asks, per allele, how well one allele
   is mirrored, the alignment asks whether the two panels organise the *variant*
   escape landscape the same way. It is the frequency-weighted linear **centred
   kernel alignment (CKA)** between the two escape-profile matrices X and Y
   (`m variants` × the alleles of each panel), each allele profile mean-centred
   across variants:

   ```
   alignment = ‖Yᵀ X‖_F²  /  (‖Xᵀ X‖_F · ‖Yᵀ Y‖_F)
   ```

   Swapping X and Y leaves this unchanged (`‖YᵀX‖_F = ‖XᵀY‖_F`), which is why it
   is symmetric.

   Carrier frequencies weight it by scaling each allele column by √w, so common
   alleles dominate the geometry. The alignment is in [0, 1], **symmetric**,
   needs no allele correspondence between panels (invariant to rotation /
   relabelling of the allele basis) and is defined for panels of different sizes
   — so alignments for different panel pairs (BoLA↔HLA, BoLA↔SLA, HLA-pop↔HLA-pop)
   are directly comparable.

   Two further steps make the number interpretable. Each panel is **centred
   twice** — once per allele across variants, once per variant across the
   panel's own weighted alleles — because two panels scoring the same peptides
   agree closely on which variants are disruptive at all, and that agreement
   reflects the peptides rather than the alleles. What remains is the
   allele-by-variant interaction. The result is then scaled by a **ceiling**:
   the alignment reached when both sides describe the same repertoire, obtained
   by splitting the reference panel into a query-sized part and a remainder,
   matched on weight profile as well as on allele count. The ceiling is a scale
   rather than a bound, so a pairing can exceed 100%. Uncertainty is a
   percentile bootstrap over variants.

   A permutation null is **not** used: permuting one panel's variant rows
   destroys all variant-level correspondence, so with tens of thousands of
   variants every pairing exceeds every permutation and the test returns its
   floor. Skip the panel step with `--no-panel`.

   *Zero-inflation sensitivity* (`--sensitivity`): escape profiles fill
   non-binding / structurally-absent cells with 0, and a block of all-zero
   (non-binding) variant rows can inflate the alignment by letting the panels
   agree merely on *which* variants are immunologically active at all. The check
   recomputes the alignment on the informative variants only — those with peak
   |escape| above `--mask-threshold` in the panel(s) selected by `--mask-mode`
   (`either`/`both`/`query`/`reference`) — and reports the difference. A small Δ
   means the shared zeros were not driving the score. (Masking is at the variant
   level, where the zero-fill lives; entry-level masking is undefined for a
   global matrix metric like CKA.)

## Pipeline

NABACHD is the terminal step of CD8scape's simulation branch. It reads the
per-allele escape scores CD8scape has already written — it does not re-run any
binding predictions:

```
simulate                      every possible SNV per segment   → variants_simulated.csv
run_supertype --per-allele    per-allele escape scores          → per_allele_best_ranks.csv
nabachd                       neighbourhood scores + figures
```

Use `run_supertype` (a whole allele panel weighted by population frequency), not
`run` (a single individual's genotype): the method needs a panel of alleles on
each side. CD8scape produces one run per segment, so a whole-genome panel is the
list of that panel's `per_allele_best_ranks.csv` files, which NABACHD combines
(you pass them, or their folders, to `--query` / `--reference`).

## Input: raw CD8scape output

The escape profile is CD8scape's per-allele output,
`per_allele_best_ranks.csv` (written by `run` / `run_supertype --per-allele`),
with columns `Frame, Locus, Mutation, MHC, ELBR_A, ELBR_D, foldchange_BR,
log2_foldchange_BR`. Variants are keyed on **(Frame, Locus, Mutation)**, the
allele is **MHC**, and the escape metric is **log2_foldchange_BR** (the same
value your `combined_simulation.csv` held). One file is produced per CD8scape
run — e.g. one per genome segment — so a whole-genome panel is a **list** of
these files, combined internally on the shared variant keys.

Reference frequencies and panel restrictions are CSVs keyed by allele name; HLA
`*` separators (`HLA-A*0101`) are reconciled with netMHCpan-style names
(`HLA-A0101`) automatically.

## Usage (command line)

```
./nabachd.jl neighbourhood <folder_path> --query <list> --reference <list> [options]
```

`--query` / `--reference` take a comma-separated list of `per_allele_best_ranks.csv`
files — or the **run folders** that contain them (the file name is appended
automatically). Paths resolve relative to `<folder_path>`; outputs are written
into it. `./nabachd.jl --help` lists every option.

**BoLA vs HLA**, each combined from the eight per-segment CD8scape run folders
(bare folders = the BoLA runs, `*_human` = HLA). Reproduces the headline
analysis: 127 × 175, k\* = 27.

```bash
./nabachd.jl neighbourhood ~/Documents/PhD/Cows/Clustering \
    --query     HA,MP,NA,NP,NS,PA,PB1,PB2 \
    --reference HA_human,MP_human,NA_human,NP_human,NS_human,PA_human,PB1_human,PB2_human \
    --frequencies supertype_panel.csv \
    --query-label BoLA --reference-label HLA
```

Swap the query for the swine `*_swine` folders to situate SLA against HLA, and so
on. Useful options: `--k <auto|INT>`, `--query-panel <csv>` (restrict the query
to a defined panel), `--value-col` / `--allele-col` / `--key` (if your columns
differ from CD8scape defaults), `--no-figures` / `--no-umap`, `--suffix` /
`--out`, `--no-fisher`. On first use the figure step precompiles CairoMakie and
is slow; later runs are fast.

A pre-combined **wide** table (e.g. `combined_simulation.csv`) is still supported
via `--query-format wide --reference-format wide --sim combined_simulation.csv
--query-prefix BoLA_BoLA- --reference-prefix HLA_HLA- --key Gene,Locus,Mutation`.

## Usage (Julia API)

```julia
using NABACHD              # or: include("src/NABACHD.jl"); using .NABACHD

# combine a list of raw per_allele_best_ranks.csv files into one panel
segs = ["HA","MP","NA","NP","NS","PA","PB1","PB2"]
query     = load_cd8scape(["$s/per_allele_best_ranks.csv"        for s in segs])
reference = load_cd8scape(["$(s)_human/per_allele_best_ranks.csv" for s in segs])

res = analyse(query, reference;
              frequencies = load_frequencies("supertype_panel.csv"),
              k = :auto)

res.k               # selected neighbourhood size (27 for this panel)
res.scores          # ranked per-allele DataFrame (weighted_nbhd_r, …)
res.calibration     # k-sweep table
res.alignment       # panel-level neighbourhood alignment (frequency-weighted
                    # CKA); `nothing` if analysed with panel=false
res.alignment.alignment            # the CKA in [0, 1]
res.alignment.n_query_effective    # alleles actually carrying weight — quote these,
res.alignment.pr_query             # and the participation ratio, not catalogue size
res.alignment.z, res.alignment.p   # permutation null — see the warning below

using CairoMakie            # triggers the figure extension
save_all_figures(res, query, reference; outdir="figures")
```

`load_cd8scape` defaults to CD8scape's columns (`Frame,Locus,Mutation` / `MHC` /
`log2_foldchange_BR`); `load_long` is the general multi-file long loader and
`load_wide` reads a wide table. Other key functions: `load_frequencies`,
`read_allele_list`, `subset_alleles`, `analyse`, and the lower-level
`correlation_matrix`, `find_k_star`, `neighbourhood_scores`,
`weighted_neighbourhood_r`. The panel-level metric is `panel_alignment` (which
also has a matrix method) and the bare kernel-alignment kernel is `cka(Mq, Mr;
wq, wr)`; `column_center` exposes the centring step. The zero-inflation check is
`alignment_sensitivity` (matrix + `ProfileSet` methods), with `informative_variants`
for the variant mask.

## Output

`neighbourhood_scores[_suffix].csv` columns:

| column | meaning |
|--------|---------|
| `allele` | query allele |
| `k` | neighbourhood size used |
| `peak_r` | max correlation with any single reference allele |
| `mean_r` | mean correlation across all reference alleles |
| `nbhd_r` | unweighted top-*k* mean correlation |
| `weighted_nbhd_r` | frequency-weighted, Fisher-z top-*k* mean — **primary metric** |
| `nearest_reference` / `nearest_r` | closest single reference allele and its r |

Plus `panel_alignment[_suffix].csv` — one row with the panel-level
**neighbourhood alignment** (`alignment`) and its permutation-null significance
(`z`, `p`, `null_mean`, `null_sd`, `nperm`), whether it was frequency-`weighted`,
and the aligned panel dimensions (`n_query`, `n_reference`, `n_variants`). Written
unless `--no-panel` is given.

With `--sensitivity`, also `panel_alignment_sensitivity[_suffix].csv` — two rows
(`all` vs `informative` variants) with each `alignment`, `z`, `p`, the variant
count, `frac_retained`, and the `mask_mode` / `mask_threshold` used.

Also `neighbourhood_calibration[_suffix].csv` (the k-sweep) and, in
`neighbourhood_figures[_suffix]/`, the figure set (viridis throughout):
`k_calibration`, `k_sd`, `mean_vs_peak`, `weighted_vs_unweighted`,
`ranked_top15`, `densities_<query>`, `densities_<reference>`, `umap_species`,
`umap_scored`.

## Notes

- **Do not quote `alignment.z` or `alignment.p`.** The permutation null shuffles
  one panel's variant rows against the other, which destroys all variant-level
  correspondence. With tens of thousands of variants every pairing exceeds every
  permutation, so *p* returns its floor of 1/(nperm+1) and cannot separate
  pairings, and *z* tracks the variant count rather than the alignment (in our
  data, corr(z, n_variants) = 0.95 against corr(z, alignment) = −0.17). They are
  retained in the struct for backwards compatibility. Use a bootstrap interval
  and the ceiling instead.
- **`n_query` / `n_reference` are catalogue sizes**, which is what the shared
  fractions are taken against. A zero-weight allele contributes to no inner
  product, so the sizes the statistic actually saw are `n_query_effective` and
  `n_reference_effective`, with `pr_query` / `pr_reference` giving the effective
  number of alleles once weight concentration is accounted for.
- **UMAP** (`umap_species`, `umap_scored`) is a stochastic 2-D visualisation
  only; layouts differ between implementations (this uses `UMAP.jl`, not R's
  uwot) and none of the neighbourhood statistics depend on it. Its random seed
  is settable with `--seed` and defaults to **1320**, matching CD8scape's default.
- `data/` and `results/` are git-ignored — CD8scape inputs and analysis outputs
  are not tracked.

## Tests

```julia
using Pkg; Pkg.test("NABACHD")
```

or, from a clone:

```bash
julia --project=. test/runtests.jl
```

## License

GPL-3.0 (see `LICENSE`).
