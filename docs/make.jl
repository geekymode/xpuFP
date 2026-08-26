using Documenter
using xpuFP

DocMeta.setdocmeta!(xpuFP, :DocTestSetup, :(using xpuFP); recursive=true)

makedocs(
    sitename = "xpuFP.jl",
    authors  = "rethna",
    modules  = [xpuFP],
    format   = Documenter.HTML(
        prettyurls   = get(ENV, "CI", "false") == "true",
        canonical    = "https://geekymode.github.io/xpuFP",
        assets       = String[],
        sidebar_sitename = true,
        edit_link    = "main",
    ),
    pages = [
        "Home"           => "index.md",
        "Formats"        => "formats.md",
        "Arithmetic"     => "arithmetic.md",
        "Block formats"  => "blocks.md",
        "Error analysis" => "analysis.md",
        "Improved FP4"   => "improved.md",
        "H·XPFP4-32"     => "xpfp4.md",
        "MSE-optimal K=16" => "mseopt16.md",
        "Redundant systems" => "redundant.md",
        "Arithmetic algorithms" => "algorithms.md",
        "Universal conversion" => "conversion_matrix.md",
        "Conversions"    => "conversions.md",
        "Hardware"       => "hardware.md",
        "HLS"            => "hls.md",
        "Examples"       => "examples.md",
        "Cost model"     => "costmodel.md",
        "Figure gallery" => "figures.md",
        "API reference"  => ["api.md",
                             "api_formats.md", "api_arith.md", "api_blocks.md",
                             "api_analysis.md", "api_redundant.md",
                             "api_convert.md", "api_viz.md"],
    ],
    doctest  = false,   # enabled once every docstring example is finalised
    warnonly = true,
)

deploydocs(
    repo         = "github.com/geekymode/xpuFP.git",
    devbranch    = "main",
    push_preview = false,
)
