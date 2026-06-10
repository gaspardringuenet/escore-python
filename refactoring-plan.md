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

EDIT: although numpy arrays are the go-to format for saving echotypes, the `sklearn`-inspired classification pipeline will require formatting the training set as tabular data:

| echotype_id: int | class: object | channel_0_Sv: float | channel_1_Sv: float | ... | channel_n_Sv: float |
|------------------|---------------|---------------------|---------------------|-----|---------------------|
| 1                | CRU           | -77.1               | -44.2               | ... | -88.9               |
| ...              | ...           | ...                 | ...                 | ... | ...                 |

### Classification pipeline

The Escore method is a semi-supervised method: echotypes are clustered into *echoclasses* in a unsupervised manner, and the echoclasses are used as target variable to learn the Escore parameters at the sample level.

For this reason, we are constrained to split the analysis into 2 separate pipelines: (1) echotypes -> echoclasses ; (2) escore.

```python
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering

set_config(enable_metadata_routing=True)

# Load data
echotypes = pd.read_csv("echotypes.csv")
X = echotypes.drop(columns=["echotype_id", "class"])    # "class" is user-defined
echotype_ids = echotypes["echotype_id"]

# STEP 1 - GROUP ECHOTYPES INTO ECHOCLASSES
# Option A - To use user-defined classes
echoclasses = echotypes["class"]

# Option B (Pipeline 1) - Learning Echoclasses
echoclassification_model = Pipeline([
    ("preprocessor", DeltaSvTransformer()),             # MVBS -> R(f) (n_samples, n_channels)
    ("aggregator", (                                    # samples -> echotype metrics (n_echotypes, n_new_features)
        EchotypesFeatureExtractor(...)
        .set_transform_request(groups=True)             # use metadata routing to pass echotype_ids
    )),  
    ("classifier", AgglomerativeClustering(...)),       # unsupervised clustering -> (n_echotypes,)
])
echotype_labels = echoclassification_model.fit_predict(X, groups=echotype_ids)

# Propagate echoclasses to Pipeline 2
unique_ids = echotypes["echotype_id"].unique()          # order matches fit_predict output
label_map = dict(zip(unique_ids, echotype_labels))
echoclasses = echotype_ids.map(label_map)

# STEP 2 - ESCORE MODEL (Pipeline 2)
escore_model = Pipeline([
    ("preprocessor", DeltaSvTransformer(...)),          # MVBS -> R(f) (n_samples, n_channels)
    ("classifier", EscoreClassifier(...)),              # supervised
])
escore_model.fit(
    X,
    y=echoclasses,                      # alternatively .fit(X, echotypes["class"]) for user-defined classes
    classifier__groups=echotype_ids,    # used to compute weights and allow equal contribution of echotypes
) 
```

Prediction should be straightforward:

```python
# Predict
## Case 1 - Predict on np.ndarray or pd.DataFrame with shape (n_samples, n_channels)
preds = escore_model.predict(X) # might require reshaping

## Case 2 - Predict on acoustic xr.DataArray (potentially Dask-backed)
preds: xr.DataArray = xr.apply_ufunc(
    escore_model.predict,
    ds_MVBS.Sv,
    input_core_dims=[['channel']],
    output_core_dims=[[]],
    dask='parallelized',
    output_dtypes=[int]
)
```

Since the Escore method is a point-wise classification, one can use `xarray.apply_ufunc` to easily predict on acoustic data while preserving coordinates and dimensions, and automatically parallelize with Dask.

### Evaluation

Using `sklearn.pipeline.Pipeline`'s means enabling cross-validation and hyper-parameter search for both steps.

Warning - cross validation should preserve echotypes grouping. Using `GroupKFold` is a solution:

```python
from sklearn.model_selection import GroupKFold, GridSearchCV

cv = GroupKFold(n_splits=5)

grid_search = GridSearch(
    escore_model,
    param_grid={"classifier__absolute_thresh": [0.7, 0.8, 0.9]},
    cv=cv
)
grid_search.fit(X, y, groups=echotype_ids)
```

Warning - current proposal does not allow combined CV accross echotypes for both `Pipeline`'s. As far as I know, this requires chaining the pipelines, which is not possible here. Separate validation of both pipeline will be required.

#### Scores

#### Echoclassification pipeline

**Question:** Do unsupervised estimators support the `score` method? (apparently not) Can unsupervised metrics be used for CV?

**Answer:** Yes, as long as we define a scorer with signature `estimator, X -> score` (such as one using the silhouette score) and pass it as the `scoring` kwarg.

**However, CV requires a `predict` method**, which not all estimators have. `AgglomerativeClustering` doesn't for instance, as it can only predict on its training set (transductive estimator). It only possesses a `fit_predict` method. One will have to accept manual evaluation switch to an inductive clusterer (which is very easy in this framework so no problem).

#### Escore pipeline

Using echoclasses as target for the escore pipeline provides straightforward supervised classification metrics (which can probably be inherited from `sklearn`).

Additional unsupervised metrics would be welcome to better caracterize the model's performance. Ideas:

- Ellipses / Gaussian distributions overlap (indicates poor separation between classes)
- ...
