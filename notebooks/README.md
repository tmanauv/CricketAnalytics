# `notebooks/`

`SDP.ipynb` is the original exploratory notebook the `cricket_analytics`
package was extracted from. It is kept unchanged for reference; the package
and the scripts in `scripts/` are the source of truth for the pipeline.

The notebook reads `Cricket Data Set.xlsx` from its working directory, so run
it from a directory containing the workbook, and install its extra
dependencies with `pip install -e ".[dl,notebook]"` (or `pip install -r requirements.txt`).
