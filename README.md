# Python implementation of the Escore algorithm

The Escore algorithm is described by Annasawmy et al. (2024). DOI - https://doi.org/10.1371/journal.pone.0309840

## Installation guide

```bash
$ conda env create --prefix ./env --file environment.yml
$ conda activate ./env
```

## Quick tour

```bash
├── data                            # data folders - configs create subfolders in those 3
│   ├── echogram_images             # echogram images (and labelme JSON) folder
│   ├── input                       # input folder (place .nc here)
│   └── interim                     # 'work' folder
├── environment.yml                 # conda environment config file
├── escore                          # tools package
│   ├── __init__.py
│   ├── __pycache__
│   ├── apps                        # interactive Dash apps
│   ├── builder.py
│   ├── config.py
│   ├── io.py
│   ├── registry.py
│   ├── test
│   └── visualize.py
├── pyproject.toml
└── scripts                         # scripts corresponding to the steps of the analysis
    ├── 00_build_image_dataset.py
    ├── 01_label_ROIs.py
    ├── 02_extract_echotypes.py
    ├── config.yml                  # default config file (my own)
    ├── config_empty.yml            # to be added
    ├── config_test_from_start.yml  # in writing
    └── config_test_with_ROIs.yml   # to be added
```

## Minimal launch on test data

### 1. Method from start (ROI labelling)

To enable ROI labelling on echogram images with [`labelme`](https://github.com/wkentaro/labelme), an image dataset must be build out of the netCDF acoustic data.

```bash
$ python scripts/00_build_image_dataset.py --config scripts/config_test_from_start.yml
```

In order to open an interactive labelling window, the user must call the second script:

```bash
$ python scripts/01_label_ROIs.py --config scripts/config_test_from_start.yml
```

For now, this step requires the path to the image folder to be correctly specified in the `session.images_dir` field of the configuration file.

No special labelme "label" is required. After creating an ROI shape, the user can save it in labelme (with `CRTL+S`). When no previous shape has been saved for this image[^1], a saving window opens. In that case *save the JSON file in the propose folder*.

[^1]: The user can also create a new ROI library on the same image dataset by changing the `session.name` field in `scripts/config_test_from_start.yml`.


### 2. Extract echo-types from ROI's

This step is performed using a [Dash](https://dash.plotly.com/) app. Running the third script launches the app. It is accessed through the web browser.

```bash
$ python scripts/02_extract_echotypes.py 
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app '02_extract_echotypes'
 * Debug mode: on
```

A `--config` keywords lets you specify the config.yml file path, as for previous steps.

1.  If you are testing the method from the start you `scripts/config_test_from_start.yml`.
2.  If you want to test extracting echo-types from pre-defined ROI on ABRAÇOS II test data, use `scripts/config_test_with_ROIs.yml`

This step is still in development.


### 3... In development

- Cluster echo-types into acoustic classes
- Define the Escore model
- Predict

## Try it on you own data

...

## Progress tracking

- [x] Handle parameters with a `config.yml` file.
- [x] ROI labelling module
  - [x] Creating a dataset of pre-printed RGB images (`scripts/00_build_image_dataset.py`)
  - [x] Label using `labelme` (`scripts/01_label_ROIs.py`)
    - [x] Track ROI shapes in a database with `SQLite`
- [ ] Echo-type extraction module (Plotly / Dash app.)
  - [x] Visualize `labelme` shapes.
    - [x] Bug : when padding==0, last row and column of mask are not shown.
  - [ ] Perform K-means clustering
    - [ ] Clustering of frequency response (r(f))
    - [ ] 
    - [ ] Handle number of channels available
      - [ ] For ROI in one frequency validity range
      - [ ] Idem for overlapping ROIs
    - [ ] Select cluster on click
      - [x] Clustering to label the Sv DataArray.
      - [ ] Create callback
    - [ ] Save echo-type data 
      - [ ] `xr.Dataset` (or Array) or new layer(s) on the `sv` `xr.DataArray`
      - [ ] Manage echo-types references in the `SQLite` database.
- [ ] Echo-type validation
- [ ] Echo-types clustering
  - [ ] FDA
  - [ ] Hierarchical clustering
- [ ] Escore computation
  - [ ] ...
- [ ] Prediction on 