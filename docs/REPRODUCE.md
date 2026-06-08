# Reproduction Notes

This file records the current reproducible workflow. All paths assume the repository root is

```text
C:\Users\gjy20\Desktop\26sp\dl\assignment_10\stage2
```

## Environment

Python executable

```powershell
C:\Users\gjy20\.conda\envs\exp3\python.exe
```

Observed runtime

```text
PyTorch 2.5.1+cu121
CUDA runtime 12.1
GPU NVIDIA GeForce RTX 4060 Laptop GPU
gsplat 1.5.3
NumPy 2.4.0
Pillow 12.0.0
scikit-image 0.26.0
```

The local CUDA toolkit used for extension compilation is

```text
C:\Users\gjy20\Desktop\26sp\dl\assignment_10\stage1\tools\cuda-12.1-local
```

The Visual Studio developer command used for CUDA extension builds is

```text
C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat
```

## Data Preparation

The shared dataset is stored in

```text
data
```

Prepared sequence variants are under

```text
work\Sequence_xx\variant_name
```

The preparation script can be rerun with

```powershell
C:\Users\gjy20\.conda\envs\exp3\python.exe scripts\prepare_stage2_sequence.py
```

Partitioned variants are built with

```powershell
C:\Users\gjy20\.conda\envs\exp3\python.exe scripts\prepare_stage2_partition.py
```

## COLMAP Reconstruction

Run robust COLMAP for one prepared variant with

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_colmap.ps1 -Sequence Sequence_04 -Variant partition_best -Matcher exhaustive -Profile robust
```

The script performs SIFT feature extraction, feature matching, mapper reconstruction, undistortion, and sparse model text conversion. The output is used by the controlled 3DGS training script.

## Controlled 3DGS Training

Run a main training job with

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_training.ps1 -Sequence Sequence_04 -Variant partition_best -OutputVariant partition_best_finetune -Iterations 24000 -MaxSide 960 -DensifyUntil 9000 -DensifyEvery 400 -DensifyFraction 0.12 -DensifyMaxNew 2200 -LambdaSsim 0.15 -LambdaEdge 0.03 -LambdaOpacity 0.0005 -SaveRenderCount 24
```

Resume from a checkpoint with

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_training.ps1 -Sequence Sequence_04 -SourceVariant partition_best -OutputVariant partition_best_finetune -ResumeCheckpoint work\Sequence_04\partition_best\output\model_final.pt -Iterations 10000 -LrScale 0.35 -SaveRenderCount 24
```

The training script writes model parameters, SuperSplat compatible PLY files, selected renders, comparisons, metrics.json, metrics_table.csv, and training curves.

## Evaluation

Image metrics for arbitrary render and ground truth directories can be computed with

```powershell
C:\Users\gjy20\.conda\envs\exp3\python.exe scripts\evaluate_image_dirs.py --renders-dir path\to\renders --gt-dir path\to\gt --output-dir path\to\evaluation
```

Geometry accuracy against the reference point clouds is computed with

```powershell
C:\Users\gjy20\.conda\envs\exp3\python.exe scripts\evaluate_stage2_geometry.py
```

Local accuracy measurements are generated with

```powershell
C:\Users\gjy20\.conda\envs\exp3\python.exe scripts\make_local_accuracy_measurements.py
```

Final report and beamer figures are regenerated with

```powershell
C:\Users\gjy20\.conda\envs\exp3\python.exe scripts\make_stage2_final_assets.py
```

## Advanced Methods

Mip Splatting source and outputs are under

```text
external\mip-splatting-main
work\Sequence_04\mip_splatting
```

2DGS source and outputs are under

```text
external\2d-gaussian-splatting-main
work\Sequence_04\twodgs
```

Depth RegularizedGS source and outputs are under

```text
external\DepthRegularizedGS-main
work\Sequence_04\depth_regularized_full
```

Depth RegularizedGS was run with sparse depth fallback enabled through

```powershell
$env:DRGS_SKIP_ZOE = '1'
```

The local compatibility package for PyTorch3D functions is stored inside

```text
external\DepthRegularizedGS-main\pytorch3d
```

This avoids changing the working PyTorch 2.5.1 CUDA 12.1 environment to a different PyTorch version.

## Document Build

Compile the report with

```powershell
xelatex -interaction=nonstopmode report.tex
xelatex -interaction=nonstopmode report.tex
```

Compile the beamer slides with

```powershell
xelatex -interaction=nonstopmode beamer.tex
xelatex -interaction=nonstopmode beamer.tex
```
