"""Reusable scaffold guidance snippets for generated experiment code."""

GENERIC_EXPERIMENT_GUIDANCE = """\
Scaffold candidate: generic experiment

Use this only if no more specific scaffold fits the task.
- Inspect available input files at runtime; do not assume hidden files exist.
- Build a minimal reliable baseline before adding research-specific modules.
- Implement AI_SCIENTIST_SMOKE_TEST mode that finishes quickly, validates one tiny data/model path, saves working/experiment_data.npy, prints SMOKE_TEST_PASS, and exits before full training.
- Save all final plottable metrics, losses, and metadata to working/experiment_data.npy.
- Prefer simple dependencies already verified by the runtime context.
- If real public data is unavailable, fail clearly unless the task explicitly permits synthetic-only validation.
"""


TIMESERIES_GUIDANCE = """\
Scaffold candidate: generic time-series experiment

Use this only for temporal, sequential, forecasting, sensor, ECG/EEG, or sliding-window tasks.
- Detect csv/parquet/npy/npz/pt files under input/ at runtime.
- Identify timestamp/order columns or tensor sequence axes; do not assume one fixed layout.
- Split chronologically when forecasting or temporal leakage matters; otherwise use a documented train/validation split.
- Fit normalization/statistics on the train split only.
- Create a small Dataset that yields windows shaped consistently, e.g. [batch, time, channels].
- Start with a compact GRU/TCN/TransformerEncoder baseline depending on the task.
- Choose metrics matching the task: MAE/RMSE for forecasting/regression, accuracy/F1/AUROC for classification.
- In AI_SCIENTIST_SMOKE_TEST mode, load a tiny slice/window, run one forward or train step, save working/experiment_data.npy, print SMOKE_TEST_PASS, and exit.
"""


TABULAR_GUIDANCE = """\
Scaffold candidate: generic tabular experiment

Use this only for structured csv/parquet/table tasks.
- Detect csv/parquet files under input/ and infer candidate target columns conservatively from the research request.
- Separate numeric/categorical features; impute missing values using train-only statistics.
- Use a clear train/validation split and avoid leakage from identifiers, target-derived columns, or future columns.
- Start with sklearn baselines or a compact PyTorch MLP only if needed.
- Save per-dataset metrics and losses to working/experiment_data.npy.
- In AI_SCIENTIST_SMOKE_TEST mode, load a small row subset, run preprocessing and one fit/forward step, save working/experiment_data.npy, print SMOKE_TEST_PASS, and exit.
"""


IMAGE_CLASSIFICATION_GUIDANCE = """\
Scaffold candidate: generic image classification experiment

Use this only for image-level classification tasks.
- Detect image folders under input/ and infer labels only from explicit directory structure or metadata.
- Use PIL/torchvision basics unless the runtime context verifies stronger dependencies.
- Resize images consistently, normalize tensors, and keep train/validation transforms separate.
- Start with a compact CNN or torchvision model only if available.
- Report accuracy/F1/AUROC as appropriate and save working/experiment_data.npy.
- In AI_SCIENTIST_SMOKE_TEST mode, load one tiny image batch, run one forward/train step, save working/experiment_data.npy, print SMOKE_TEST_PASS, and exit.
"""


SEGMENTATION_GUIDANCE = """\
Scaffold candidate: generic image segmentation experiment

Use this for image/mask segmentation tasks when no more specific segmentation scaffold fits.
- Detect image/mask pairs under input/ using observed directory names; do not invent paths.
- Verify each image has a matching mask and resize image/mask consistently.
- Convert masks to binary or multiclass targets according to observed masks and task description.
- Start with a compact UNet-like model or simple encoder-decoder baseline.
- Use Dice/IoU plus loss curves as primary metrics.
- Save per-dataset metrics, losses, and small prediction summaries to working/experiment_data.npy.
- In AI_SCIENTIST_SMOKE_TEST mode, load one image/mask pair, run one forward/train step, save working/experiment_data.npy, print SMOKE_TEST_PASS, and exit.
"""


POLYP_SEGMENTATION_GUIDANCE = """\
Scaffold candidate: polyp segmentation

Use this only when the task or observed input explicitly mentions polyp, Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, ETIS, or related colonoscopy segmentation data.
- Prefer prepared paths only if observed: input/Kvasir-SEG/images, input/Kvasir-SEG/masks, input/CVC-ClinicDB/images, input/CVC-ClinicDB/masks.
- If those paths are not observed, inspect input/ and fail clearly rather than inventing paths.
- Pair RGB endoscopy images with binary masks, resize both consistently, and ensure mask targets are [B, 1, H, W] float or long as required by the loss.
- Use Dice and IoU for source/target or per-dataset reporting.
- Keep cross-domain validation explicit, e.g. train on Kvasir-SEG and evaluate CVC-ClinicDB when both are observed.
- In AI_SCIENTIST_SMOKE_TEST mode, load one real image/mask pair, run one tiny forward/train step, save working/experiment_data.npy, print SMOKE_TEST_PASS, and exit.
"""
