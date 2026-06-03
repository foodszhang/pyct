# Real Phantom FMT Workflow

## Goal

Use CT to define the physical phantom space, use MCX to generate seven-view simulated
training pairs, and use processed real fluorescence images only for test-time
reconstruction and simulation-quality comparison.

Do not create fake `gt_voxels.npy` for a real acquisition.

## Current Bundle

Bundle root:

```text
/home/foods/pro/FMT-SimGen/data/real_phantom_3gy_ft_1
```

Important files:

```text
subjects/phantom_3gy_ft_1/shared/body_labels.nii.gz
subjects/phantom_3gy_ft_1/shared/atlas_labels.npz
subjects/phantom_3gy_ft_1/shared/mcx_volume_trunk.bin
subjects/phantom_3gy_ft_1/shared/mesh.npz
subjects/phantom_3gy_ft_1/shared/frame_manifest.json
subjects/phantom_3gy_ft_1/shared/view_config.json
samples/sample_phantom_3gy_ft_1/proj.npz
samples/sample_phantom_3gy_ft_1/proj_real_processed_full.npz
samples/sample_phantom_3gy_ft_1/proj_real_corrected_full.npz
samples/sample_phantom_3gy_ft_1/projection_preprocess.json
```

`proj.npz` is the model input: seven angles `[-90,-60,-30,0,30,60,90]`, each
`256x256`. The two `*_full.npz` archives keep all 19 real angles for QA and future
preprocessing changes.

## Reflection Correction

The right-side reflection is partly fixed in camera coordinates while real sources
move with turntable angle. The default correction subtracts the pixelwise 20th
percentile across all 19 angles and clips negative values to zero.

A hard right-side mask is intentionally not applied: it removes real source signal
when a source rotates through the right side.

Review these PNG files after each acquisition:

```text
proj_real_before.png
proj_real_corrected.png
reflection_static_background.png
proj_model_7view.png
```

## Repeatable Commands

Import a CT reconstruction and real fluorescence acquisition:

```bash
.venv/bin/python scripts/import_real_zj_to_fmt_simgen.py \
  --subject-id phantom_3gy_ft_1 \
  --ct recon_output/20260602-3gy-ft-1-vshift-512-zm60/rec_vshift_hamming.nii.gz \
  --ct-voxel-size-mm 0.3 \
  --flu-npz-dir 20260602-1101-1000-3gy-ft/npz \
  --output-root /home/foods/pro/FMT-SimGen/data/real_phantom_3gy_ft_1
```

Build MCX volume, coarse FEM mesh, and view configuration:

```bash
.venv/bin/python scripts/build_real_phantom_fmt_assets.py \
  --bundle-root /home/foods/pro/FMT-SimGen/data/real_phantom_3gy_ft_1 \
  --subject-id phantom_3gy_ft_1
```

Estimate an editable source prior, run Windows MCX, and score one candidate:

```bash
python scripts/estimate_real_phantom_gt.py \
  --sample-dir /home/foods/pro/FMT-SimGen/data/real_phantom_3gy_ft_1/samples/sample_phantom_3gy_ft_1 \
  --shared-dir /home/foods/pro/FMT-SimGen/data/real_phantom_3gy_ft_1/subjects/phantom_3gy_ft_1/shared

/home/foods/pro/FMT-SimGen/.venv/bin/python scripts/run_real_phantom_mcx_candidate.py \
  --sample-dir /home/foods/pro/FMT-SimGen/data/real_phantom_3gy_ft_1/samples/sample_phantom_3gy_ft_1 \
  --shared-dir /home/foods/pro/FMT-SimGen/data/real_phantom_3gy_ft_1/subjects/phantom_3gy_ft_1/shared \
  --mcx-exe /mnt/f/win-pro/bin/mcx.exe
```

Each source hypothesis is staged under `/mnt/f/win-pro/mcx_runs` using a parameter
hash. MCX fluence, seven-view projections, and `projection_similarity.json` remain
isolated per candidate.

The currently selected prior remains explicitly estimated:

```text
samples/sample_phantom_3gy_ft_1/tumor_params_fitted_estimated.json
samples/sample_phantom_3gy_ft_1/gt_voxels_fitted_estimated.npy
samples/sample_phantom_3gy_ft_1/gt_nodes_fitted_estimated.npy
samples/sample_phantom_3gy_ft_1/fitted_estimated_gt_manifest.json
```

## Downstream Status

### GISC-FMT

Training remains the existing simulated path:

```text
MCX source volume -> simulated proj.npz + gt_voxels.npy -> GISC-FMT training
```

The real bundle already supports projection-only proposal generation. The existing
Lightning test dataset still requires `gt_voxels.npy`, so real inference needs a
separate prediction entrypoint that builds an inference grid from
`frame_manifest.json` and saves reconstruction without GT metrics.

### DU2Vox

Stage 2 can already read the real seven-view `proj.npz`. Stage 1 still needs
`measurement_b.npy`, a surface-node measurement vector. For real inference, add a
bridge that projects visible surface nodes into each corrected real view and robustly
fuses sampled intensities into `measurement_b.npy`.

The bridge must use `frame_manifest.json`, `view_config.json`, and the same real-camera
crop recorded in `projection_preprocess.json`.

## Manual Review Still Required

The real-camera crop center is currently provisional:

```text
center_uv_px = [392, 312]
crop = [240, 160, 544, 464]
camera_pixel_size_mm = 0.26300825
```

Confirm `proj_model_7view.png` visually before formal simulation fitting. A blank
phantom acquisition with the same exposure and camera position is useful but not
required. When a blank acquisition is not practical, retain static percentile
background subtraction and treat the residual right-side reflection as an explicit
low-confidence fitting region.

## Mouse CT Batch

Inventory the existing mouse acquisitions:

```bash
python scripts/inventory_true_data_mice.py
python scripts/cbct_qc.py --root /home/foods/pro/true_data/ZJ --skip-mask
python scripts/rank_true_data_mice.py
```

Generate Digimouse-like approximate tissue labels after reconstructing one mouse CT:

```bash
python scripts/segment_mouse_ct_approx.py \
  --volume /home/foods/pro/true_data/ZJ/13/rec_pyct_final_fov480_ramlak.nii.gz \
  --output-dir /home/foods/pro/true_data/ZJ/13/segmentation_approx_v3 \
  --head-axis x --head-side high \
  --roi-fraction 0.03 0.92 0.12 0.88 0.04 0.48
```

Body and bone labels are CT-supported. Major-organ labels are smooth anatomical
priors clipped to the body ROI; they are not manual or contrast-supported
segmentations.

## Real Mouse Pilot

`ZJ/7` is imported as a held-out real inference sample. Its CT and fluorescence
must not enter training or validation labels:

```text
/home/foods/pro/FMT-SimGen/data/real_mouse_zj/splits/train_simulated.txt
/home/foods/pro/FMT-SimGen/data/real_mouse_zj/splits/val_simulated.txt
/home/foods/pro/FMT-SimGen/data/real_mouse_zj/splits/inference.txt
```

The old mouse acquisition uses the reversed turntable sequence and its own camera
crop calibration:

```text
angles = [90, 80, ..., -80, -90]
center_uv_px = [240, 244]
camera_pixel_size_mm = 0.12
```

The selected `ZJ/7` approximate segmentation is:

```text
/home/foods/pro/true_data/ZJ/7/segmentation_approx_smooth/digimouse_like_labels.nii.gz
```

Build smooth multi-tissue MCX/FEM assets and a small simulated pilot split:

```bash
/home/foods/pro/FMT-SimGen/.venv/bin/python scripts/build_real_phantom_fmt_assets.py \
  --bundle-root /home/foods/pro/FMT-SimGen/data/real_mouse_zj \
  --subject-id mouse_zj_7 --mesh-downsample 2

python scripts/generate_mouse_simulated_samples.py \
  --bundle-root /home/foods/pro/FMT-SimGen/data/real_mouse_zj \
  --subject-id mouse_zj_7 --count 3 --validation-count 1

/home/foods/pro/FMT-SimGen/.venv/bin/python scripts/run_mouse_mcx_samples.py \
  --bundle-root /home/foods/pro/FMT-SimGen/data/real_mouse_zj \
  --subject-id mouse_zj_7 --force
```

The real mouse GT-like arrays remain estimated priors for reconstruction testing,
not measured labels. Run local MCX fitting before exporting them:

```bash
python scripts/generate_mouse_real_local_variants.py \
  --sample-dir /home/foods/pro/FMT-SimGen/data/real_mouse_zj/samples/sample_mouse_zj_7_real_1000 \
  --shared-dir /home/foods/pro/FMT-SimGen/data/real_mouse_zj/subjects/mouse_zj_7/shared
```

Whole-image NCC is only an initial fitting signal. For the three-source phantom
and real mouse, manually mark source centers separately from reflections using:

```text
samples/sample_phantom_3gy_ft_1/manual_source_annotation_phantom.png
samples/sample_mouse_zj_7_real_1000/manual_source_annotation_mouse_zj_7.png
```

The real phantom recipe is treated separately from mouse tissue optics:

```text
45 ml water + 1 g agar powder = 2.22% w/v agar-water gel
mua  = 0.002 mm^-1
musp = 0.060 mm^-1
g    = 0.90
mus  = 0.600 mm^-1
n    = 1.334
```

These values are estimates for a near-NIR agar-water gel without a recorded added
absorber or scatterer. They replace the previous mouse soft-tissue material for
`tag=1` in the phantom MCX media file only.
