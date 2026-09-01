#!/usr/bin/env julia

###############################################################################
# nabachd.jl — NABACHD: Neighbourhood Analysis of Binding-escape Against Class-I HLA Data
#
# Command-line interface, in the style of CD8scape: call a command and point it
# at your CD8scape output. Consumes raw `per_allele_best_ranks.csv` files
# directly — pass a list and they are combined internally.
#
# USAGE:
#   ./nabachd.jl neighbourhood <folder_path> --query <list> --reference <list> [options]
#
# Example — BoLA (cattle) query vs the HLA (human) reference, each combined from
# the per-segment CD8scape run folders under <folder_path>:
#
#   ./nabachd.jl neighbourhood /path/to/runs \
#       --query     HA,MP,NA,NP,NS,PA,PB1,PB2 \
#       --reference HA_human,MP_human,NA_human,NP_human,NS_human,PA_human,PB1_human,PB2_human \
#       --frequencies supertype_panel.csv \
#       --query-label BoLA --reference-label HLA
#
# Run `./nabachd.jl --help` for the full option list.
###############################################################################

using Pkg
Pkg.activate(@__DIR__)

using CSV
using DataFrames

include(joinpath(@__DIR__, "src", "NABACHD.jl"))
using .NABACHD

# ── argument helpers (manual parsing, CD8scape style) ───────────────────────
function getflag(argv, name; default=nothing)
    i = findfirst(==(name), argv)
    (i === nothing || i == length(argv) || startswith(argv[i+1], "--")) && return default
    return argv[i+1]
end
hasflag(argv, name) = name in argv
default_strip(prefix) = (i = findfirst('_', prefix); i === nothing ? "" : prefix[1:i])
splitlist(s) = String.(strip.(split(s, ",")))

function print_help()
    println("""
NABACHD — Neighbourhood Analysis of Binding-escape Against Class-I HLA Data
The meaningfulness of MHC-I binding escape between allele panels, from CD8scape output.

USAGE:
    ./nabachd.jl neighbourhood <folder_path> --query <list> --reference <list> [options]

Computes, for every allele in a QUERY panel, the frequency-weighted top-k
neighbourhood r against a REFERENCE panel. Paths in the options are resolved
relative to <folder_path>; outputs are written into <folder_path>.

DATA (raw CD8scape output — default):
    --query <a,b,c>            Comma-separated CD8scape per-allele files (or run
                               folders) whose alleles form the QUERY panel; they
                               are combined internally.
    --reference <a,b,c>        As above, for the REFERENCE panel.
    --input-name <file>        Filename to read inside any folder entry
                               (default per_allele_best_ranks.csv).
    --value-col <name>         Escape metric column (default log2_foldchange_BR).
    --allele-col <name>        Allele column (default MHC).
    --key <c1,c2,c3>           Variant key columns (default Frame,Locus,Mutation).

DATA (wide table — alternative, e.g. a pre-combined simulation):
    --query-format wide  --reference-format wide
    --sim <file>               Wide table used for both panels, or per-panel
                               --query-file / --reference-file.
    --query-prefix <str>       Column prefix selecting query alleles (e.g. BoLA_BoLA-).
    --reference-prefix <str>   Column prefix selecting reference alleles.
    --query-strip / --reference-strip <str>   Prefix stripped to get allele names
                               (default: up to and including the first '_').
    (with wide combined_simulation.csv, pass --key Gene,Locus,Mutation)

WEIGHTS & PANELS:
    --frequencies <file>       Reference carrier-frequency CSV (weights). Omit for
                               an unweighted analysis.
    --freq-allele-col <name>   (default Allele)   --freq-value-col <name> (default Frequency)
    --query-panel <file>       Restrict the query to alleles listed in this CSV.
    --reference-panel <file>   Restrict the reference likewise.
    --panel-col <name>         Allele column in the panel CSV(s) (default Allele).

METHOD:
    --k <auto|INT>             Neighbourhood size (default auto = crossover k*).
    --no-fisher                Plain weighted mean instead of Fisher-z averaging.
    --seed <int>               Random seed for the UMAP embedding (default 1320,
                               matching CD8scape's default).

OUTPUT:
    --out <dir>                Output directory (default: <folder_path>).
    --suffix <name>            Appended to output names (…_<name>).
    --query-label <name>       Label for query in figures (default: query).
    --reference-label <name>   Label for reference in figures (default: reference).
    --top-n <int>              Alleles to rank / label in figures (default 15).
    --no-figures               Skip figure rendering.
    --no-umap                  Skip the (slow) UMAP panels.
    --figure-format <png|svg|pdf>  (default png)
    --help, -h                 Print this message.
""")
end

# resolve a folder entry to its CD8scape file (folder → folder/<input-name>)
function resolve_entry(entry, resolve, inputname)
    p = resolve(String(strip(entry)))
    isdir(p) ? joinpath(p, inputname) : p
end

# ── build one panel (query or reference) from the flags ─────────────────────
function build_side(role, argv, resolve, key_cols)
    fmt = getflag(argv, "--$role-format"; default=getflag(argv, "--format"; default="cd8scape"))

    ps = if fmt in ("cd8scape", "long")
        spec = getflag(argv, "--$role")
        spec === nothing &&
            error("--$role is required: a comma-separated list of CD8scape per-allele files or run folders")
        inputname = getflag(argv, "--input-name"; default="per_allele_best_ranks.csv")
        files = [resolve_entry(e, resolve, inputname) for e in splitlist(spec)]
        load_long(files; key_cols=key_cols,
                  allele_col=getflag(argv, "--$role-allele-col"; default=getflag(argv, "--allele-col"; default="MHC")),
                  value_col=getflag(argv, "--$role-value-col"; default=getflag(argv, "--value-col"; default="log2_foldchange_BR")))
    elseif fmt == "wide"
        fref = getflag(argv, "--$role-file"; default=getflag(argv, "--sim"))
        fref === nothing && error("wide $role needs --$role-file or --sim")
        prefix = getflag(argv, "--$role-prefix")
        cols = getflag(argv, "--$role-cols")
        strip_ = getflag(argv, "--$role-strip";
                         default = prefix === nothing ? "" : default_strip(prefix))
        load_wide(resolve(fref); key_cols=key_cols, allele_prefix=prefix,
                  allele_cols = cols === nothing ? nothing : splitlist(cols),
                  strip_prefix=strip_)
    else
        error("unknown --$role-format \"$fmt\" (expected cd8scape or wide)")
    end

    panel = getflag(argv, "--$role-panel")
    if panel !== nothing
        wanted = read_allele_list(resolve(panel);
                                  allele_col=getflag(argv, "--panel-col"; default="Allele"))
        ps = subset_alleles(ps, wanted)
    end
    return ps
end

# ── main ────────────────────────────────────────────────────────────────────
if isempty(ARGS) || ARGS[1] in ("-h", "--help")
    print_help(); exit(0)
end

command = ARGS[1]
if command != "neighbourhood"
    println("Error: unknown command \"$command\". Try --help."); exit(1)
end
if length(ARGS) < 2 || startswith(ARGS[2], "--")
    println("Error: missing <folder_path> for neighbourhood command."); exit(1)
end

folder = ARGS[2]
argv = ARGS[3:end]
resolve(p) = isabspath(p) ? p : normpath(joinpath(folder, p))
key_cols = splitlist(getflag(argv, "--key"; default="Frame,Locus,Mutation"))

println("Loading query panel ...")
query = build_side("query", argv, resolve, key_cols)
println("  ", query)
println("Loading reference panel ...")
reference = build_side("reference", argv, resolve, key_cols)
println("  ", reference)

freqs = nothing
if (ff = getflag(argv, "--frequencies")) !== nothing
    freqs = load_frequencies(resolve(ff);
                             allele_col=getflag(argv, "--freq-allele-col"; default="Allele"),
                             freq_col=getflag(argv, "--freq-value-col"; default="Frequency"))
    println("  loaded ", length(freqs), " reference frequencies")
end

kspec = getflag(argv, "--k"; default="auto")
k = kspec == "auto" ? :auto : parse(Int, kspec)
fisher = !hasflag(argv, "--no-fisher")

println("Running analysis ...")
res = analyse(query, reference; frequencies=freqs, k=k, fisher=fisher)
println("  neighbourhood size k = ", res.k)

suffix = getflag(argv, "--suffix"; default="")
sfx = isempty(suffix) ? "" : "_" * suffix
outdir = resolve(getflag(argv, "--out"; default="."))
mkpath(outdir)
scores_path = joinpath(outdir, "neighbourhood_scores$sfx.csv")
calib_path  = joinpath(outdir, "neighbourhood_calibration$sfx.csv")
CSV.write(scores_path, res.scores)
CSV.write(calib_path, res.calibration)
println("  wrote ", scores_path)
println("  wrote ", calib_path)

if !hasflag(argv, "--no-figures")
    println("Rendering figures ...")
    include(joinpath(@__DIR__, "src", "plots.jl"))
    figdir = joinpath(outdir, "neighbourhood_figures$sfx")
    paths = NeighbourhoodPlots.save_all_figures(res, query, reference;
                outdir=figdir,
                query_label=getflag(argv, "--query-label"; default="query"),
                reference_label=getflag(argv, "--reference-label"; default="reference"),
                top_n=parse(Int, getflag(argv, "--top-n"; default="15")),
                umap=!hasflag(argv, "--no-umap"),
                fmt=getflag(argv, "--figure-format"; default="png"),
                seed=parse(Int, getflag(argv, "--seed"; default="1320")))
    println("  wrote ", length(paths), " figures → ", figdir)
end
println("Done.")
