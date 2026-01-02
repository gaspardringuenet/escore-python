# Python implementation of the Escore algorithm

The Escore algorithm is described by Annasawmy et al. (2024). DOI - https://doi.org/10.1371/journal.pone.0309840

## Progress tracking

- [x] Handle parameters with a `config.yml` file.
- [x] ROI labelling module
  - [x] Creating a dataset of pre-printed RGB images (`scripts/00_build_image_dataset.py`)
  - [x] Label using `labelme` (`scripts/01_label_ROIs.py`)
    - [ ] Track ROI shapes in a database with `SQLite`
- [ ] Echo-type extraction module (Plotly / Dash app.)
  - [x] Visualize `labelme` shapes.
  - [ ] Perform K-means clustering
    - [ ] Make sure we cluster the right features (vector of Sv or r(f))
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