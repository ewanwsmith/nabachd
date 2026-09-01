# NABACHD

**N**eighbourhood **A**nalysis of **B**inding-escape **A**gainst **C**lass-I
**H**LA **D**ata

*nàbachd* /ˈnaːbəxk/ — neighbourhood in Scottish Gaelic

A Julia implementation of a metric of **MHC-I neighbourhoods**: a framework for
quantifying the *meaningfulness* of MHC-I binding escape from one allele panel
relative to another, from
[CD8scape](https://doi.org/10.64898/2026.04.20.719634) output.

For each allele in a **query** panel (e.g. cattle BoLA), its per-variant escape
profile is correlated against every allele in a larger, optionally
frequency-weighted **reference** panel (e.g. the human HLA repertoire). A
neighbourhood size *k* is chosen automatically, and the **frequency-weighted
top-*k* neighbourhood r** summarises how closely each query allele's escape
landscape is mirrored by the reference. Both panels are fully configurable, so
the method applies to any pair of panels netMHCpan / CD8scape can handle — a
non-human panel against the human HLA repertoire, or one human panel against
another (e.g. across populations), not only BoLA vs HLA.

It reads **raw CD8scape output directly** and, in the style of CD8scape, is
driven by a command you point at your data. You pass a list of
`per_allele_best_ranks.csv` files (or their run folders) and it combines them
internally — no pre-processing.

## Installation

Requires Julia ≥ 1.10 ([juliaup](https://github.com/JuliaLang/juliaup)
recommended).

```bash
git clone https://github.com/ewanwsmith/nabachd.git
cd nabachd
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The core module (`CSV`, `DataFrames`, `Statistics` only) loads quickly; the
plotting module additionally pulls in `CairoMakie` and `UMAP` and is loaded only
when figures are requested.

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
4. **Weighted neighbourhood r** (primary metric): for the *k* nearest reference
   alleles, weight their correlations by reference (carrier) frequency,
   renormalise within the neighbourhood, and average on the Fisher *z* scale
   (back-transformed). With no frequency file the analysis is unweighted.

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
include("src/NABACHD.jl"); using .NABACHD

# combine a list of raw per_allele_best_ranks.csv files into one panel
segs = ["HA","MP","NA","NP","NS","PA","PB1","PB2"]
query     = load_cd8scape(["$s/per_allele_best_ranks.csv"        for s in segs])
reference = load_cd8scape(["$(s)_human/per_allele_best_ranks.csv" for s in segs])

res = analyse(query, reference;
              frequencies = load_frequencies("supertype_panel.csv"),
              k = :auto)

res.k               # selected neighbourhood size (27 for this panel)
res.scores          # ranked DataFrame
res.calibration     # k-sweep table

include("src/plots.jl")     # loads CairoMakie + UMAP
NeighbourhoodPlots.save_all_figures(res, query, reference; outdir="figures")
```

`load_cd8scape` defaults to CD8scape's columns (`Frame,Locus,Mutation` / `MHC` /
`log2_foldchange_BR`); `load_long` is the general multi-file long loader and
`load_wide` reads a wide table. Other key functions: `load_frequencies`,
`read_allele_list`, `subset_alleles`, `analyse`, and the lower-level
`correlation_matrix`, `find_k_star`, `neighbourhood_scores`,
`weighted_neighbourhood_r`.

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

Also `neighbourhood_calibration[_suffix].csv` (the k-sweep) and, in
`neighbourhood_figures[_suffix]/`, the figure set (viridis throughout):
`k_calibration`, `k_sd`, `mean_vs_peak`, `weighted_vs_unweighted`,
`ranked_top15`, `densities_<query>`, `densities_<reference>`, `umap_species`,
`umap_scored`.

## Notes

- **UMAP** (`umap_species`, `umap_scored`) is a stochastic 2-D visualisation
  only; layouts differ between implementations (this uses `UMAP.jl`, not R's
  uwot) and none of the neighbourhood statistics depend on it. Its random seed
  is settable with `--seed` and defaults to **1320**, matching CD8scape's default.
- `data/` and `results/` are git-ignored — CD8scape inputs and analysis outputs
  are not tracked.

## Tests

```bash
julia --project=. test/runtests.jl
```

## License

GPL-3.0 (see `LICENSE`).
