# Refactoring the Escore package API

**Goal:**  implement the [`scikit-learn`](https://scikit-learn.org/stable/developers/develop.html#developing-scikit-learn-estimators) estimator API.

## What the new pipeline should look like

### Interactive echotypes extraction

*echotypes* designate subsets of an echogram containing reference frequency responses ($R(f) = \Delta \text{MVBS} = \text{MVBS} - \text{MVBS}_{\text{ref. channel}}$). They are defined as having both spatio-temporal and $R(f)$ homegeneity.

In the Escore method, echotypes are first delineated manually using an echogram annotation software. Each future echotype is embedded in a region of interest (ROI) of rectangular or polygonal shape. Our workflow assumes that the ROI selection step has already been performed and that ROIs are represented in a single output file which can be parses using the [`Echoregions`](https://echoregions.readthedocs.io/en/latest/) package.

```python
import echoregions as er
import xarray as xr

# Open MVBS dataset
ds_MVBS = xr.open_dataset('acoustic_file.zarr')

# Read regions file
roi_annotations = er.Regions2D('regions.csv', input_type='CSV')
```

The next step is an iterative process:

```raw
for each ROI:
    fetch Sv data
    perform segmentation
    select the class corresponding to the echotype
    inspect
    decide (save or ignore)
    if save:
        save echotype Sv data & recipe
```

Comments:

- The segmentation task is typically a K-Means clustering of $R(f)$.
- The inspection step is especially important. Visual checks must include plotting the frequency response (mean, sd, distribution) and visualizing the selecting samples as a masked echogram. Some statistical tests could also be performed. Plots should be interactive for a better appreciation (use `hvplot`).
- For reproducibility, saving should include both the samples (as numpy array), the class (if required) and the recipe (segmentation parameters).

The echotype extraction process should be interactive and fast in a jupyter notebook.

Note: We leave this step out for now in order to focus on the modelling pipeline. Let's assume echotypes (and optionally their user-defined classes) are stored and can be acessed as an iterable of numpy arrays.

### Classification pipeline

Proposed pipeline API

```python
from sklearn.pipeline import Pipeline

from escore import EscoreClassifier
from escore.preprocessing import DeltaSvTransformer, EchoClassesClassifier

model = Pipeline([
    ("preprocessor", DeltaSvTransformer(pairs=[(1, 0), (2, 0)])),
    ("meta", EchoClassesClassifier(n_classes=7)),
    ("classifier", EscoreClassifier(...))
])
```

Where:

- "preprocessor": computes $R(f)$ for each (time, depth) sample
- "meta": unsupervised classification of the echotypes. Currently a composed of a feature extraction (e.g. mean $R(f)$) and a clustering
- "classifier": model for the classification samples

The "meta" step should only be used during fitting. Besides, it should be optional: providing user-defined classes as target for fitting should be allowed. Hence EchoClassesClassifier might not be integrated directly as a pipeline step, but rather as an optional sub-estimator of EscoreClassifier (?).

Using the pipeline should be simple:

```python
from escore.echotypes import get_data

# Fit on echotypes
data: List[np.ndarray] = get_data(...)
model.fit(data)

# Evaluate (don't know how yet)
...

# Predict
## Case 1 - Predict on np.ndarray
preds = model.predict(X) # might require reshaping

## Case 2 - Predict on acoustic xr.DataArray (potentially Dask-backed)
preds: xr.DataArray = xr.apply_ufunc(
    model.predict,
    ds_MVBS.Sv,
    input_core_dims=[['channel']],
    output_core_dims=[[]],
    vectorize=True,
    dask='parallelize',
    output_dtypes=[int]
)
```

Since the Escore method is a point-wise classification, one can use `xarray.apply_ufunc` to easily predict on acoustic data while preserving coordinates and dimensions, and automatically parallelize with Dask.

### Evaluation

#### Unsupervised metrics


#### Supervised evaluation