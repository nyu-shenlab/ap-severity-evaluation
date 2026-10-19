# ap-severity-evaluation
Predict acute pancreatitis severity from contrast CT on any dataset (inference only).

### Setup
```bash
conda create --name aps python=3.10
conda activate aps
sh setup_env.sh
cp configs/ensemble_template.yaml configs/ensemble.yaml   # then fill in your weight paths
```

### Input
A CSV with one row per axial CT slice; see `configs/datalist_template.csv`.

| column | required | meaning |
|---|---|---|
| `rid` | yes | integer patient id |
| `series_number` | yes | CT phase; all phases are scored |
| `absolute_image_path` | yes | DICOM, `.nii.gz`, or `.hdf5` |
| `index`, `RescaleIntercept`, `RescaleSlope` | HDF5 only | slice index and HU rescale |
| `view` | no | only `axial` rows are used |
| `label` | no | `MILD` / `MOD` / `SEV` |

An HDF5 series must be a single file.

### Run
```bash
sbatch gpujob_test.slurm /path/to/datalist.csv my_cohort output
```
Single model: `python evaluation.py --datalist_path <csv> --config_path <config.yaml> --pretrain_weights <ckpt>`.
Add `--phase visualization --visualization_ids <rid> ...` for GradCAM.

### Output
- `<name>_<model>.json`: per-model `[P(mild), P(moderate), P(severe)]` per `rid`.
- `<name>_ensemble.csv`: `rid, mild_pred, severe_pred` (+ labels if given).

`mild_pred` is the mean P(mild) of the 5 mild models. `severe_pred` is the mean of the
5 severe models' P(severe), each **min-max scaled over the cohort**, so it is a
within-cohort ranking, not a probability, and unreliable for small cohorts.
