# ap-severity-evaluation
Predict acute pancreatitis severity from contrast CT, on any dataset.

This repo runs inference only. It reproduces the evaluation path of the training
repo (`ap-severity`, `bin/train.py --phase evaluation --select_all_phases`) and the
10-model ensemble used in the final analysis.

### Setup environment
```bash
# on Bigpurple
conda create --name aps python=3.10
conda activate aps
sh setup_env.sh
```

### Input: datalist CSV
One row per axial CT slice.

| column | required | meaning |
|---|---|---|
| `rid` | yes | integer patient / study id; one prediction per `rid` |
| `series_number` | yes | CT phase; every phase of a patient is scored |
| `absolute_image_path` | yes | DICOM file, `.nii.gz` volume, or `.hdf5` volume |
| `index`, `RescaleIntercept`, `RescaleSlope` | HDF5 only | slice index in the volume and HU rescale |
| `view` | no | if present, only `axial` rows are used |
| `label` | no | `MILD` / `MOD` / `SEV`; only used to report labels next to predictions |

See `configs/datalist_template.csv` for a filled-in example:
- patient `1001` has two phases stored as one DICOM file per slice;
- patient `1002` has no label (leave `label` empty for unlabeled data);
- patient `1003` is an HDF5 volume (dataset `image`, shape `[slices, H, W]`, raw stored
  values), one row per slice with its `index` and the DICOM rescale values.

Use one storage format per series. An HDF5 series must be a single file: the loader
reads all of that series' `index` rows from the first row's path. Slice order does not
matter (slices are attention-pooled).
Leave `index` / `RescaleIntercept` / `RescaleSlope` empty for DICOM; they are read from the file.

Slices are windowed to HU [-155, 245], resized to the training resolution, and
standardized per image.

### Run
```bash
# from the repo root
sbatch gpujob_test.slurm /path/to/datalist.csv my_cohort output
```
or directly on a GPU node:
```bash
python evaluation.py --datalist_path datalist.csv --models_file configs/ensemble.yaml \
    --output_path output --prediction_name my_cohort
```
`configs/ensemble.yaml` lists the model directories (each with `config.yaml` and the
checkpoint). It is git-ignored: create it from the template and fill in your paths,
```bash
cp configs/ensemble_template.yaml configs/ensemble.yaml
```
or keep the paths as they are and point `--weights_root` at your copy of the weights.

A single model is run with `--config_path <model_dir>/config.yaml --pretrain_weights <ckpt>`.
GradCAM maps are `--phase visualization --visualization_ids <rid> ...` with a single model.

### Output
- `<name>_<model>.json` per model: `rids`, `predictions` rows `[P(mild), P(moderate), P(severe)]`,
  `labels` (ordinal pairs, `[-1, -1]` when unlabeled).
- `<name>_ensemble.csv`: `rid, mild_pred, severe_pred, mild_label, severe_label`.

How the ensemble is formed (same as the final analysis):
- per model, every phase is scored and each of the two ordinal logits is max-pooled
  across phases before the sigmoid;
- `mild_pred` = mean of P(mild) over the 5 mild models;
- `severe_pred` = mean over the 5 severe models of P(severe) after **min-max scaling
  each model over the scored cohort**. `severe_pred` is therefore a within-cohort
  ranking, not a probability: it changes if patients are added or removed, and is
  meaningless for very small cohorts (a warning is logged under 10 patients).

No test-time augmentation is applied.
