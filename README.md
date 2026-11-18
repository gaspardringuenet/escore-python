# Escore Python

Python re-implementation of the Escore multifrequency acoustic backscatter
classification method. It provides tools for extracting and reviewing
echotypes from regions of interest, fitting a classifier, and classifying data.

The method and original MATLAB implementation are described in: Annasawmy,
Roudaut and Lebourges Dhaussy (2024), *Impact of an eddy dipole of the
Mozambique channel on mesopelagic organisms, highlighted by multifrequency
backscatter classification*, PLOS ONE, [doi:10.1371/journal.pone.0309840](https://doi.org/10.1371/journal.pone.0309840).

## Development

Requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/). In a
checkout, install dependencies and run the tests with:

```bash
uv sync --locked
uv run --locked pytest
uv build
```

The distribution is named `escore-python`; import it as `escore`. The
`fancy-echograms` dependency is sourced from GitHub by uv in this checkout.
`uv.sources` is not included in wheel metadata or used by pip, so publishing
to PyPI or installing this package as a dependency requires that dependency
to be available from a package index or otherwise explicitly supplied.

See `notebooks/processing.ipynb` for an example workflow. Extraction snapshots
use joblib: only load snapshots from trusted sources.