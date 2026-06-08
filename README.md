# 3D Gaussian Splatting Indoor Reconstruction Stage 2

Authors

Jingyi Guo PB22011958

Siqi Han PB22010386

Yiru Wang PB22081510

This repository contains the Stage 2 indoor scene reconstruction work for Assignment 10. It includes the shared dataset, COLMAP preparation scripts, controlled 3DGS training scripts, quantitative evaluation scripts, report source, beamer source, rendered result figures, geometric accuracy evaluation, advanced method ablations, and member division notes.

## Current Results

The final controlled 3DGS variants are listed below. These are the values used in the report and beamer.

| Sequence | Final variant | Images | PSNR | SSIM | Main note |
| --- | --- | ---: | ---: | ---: | --- |
| Sequence 01 | enhanced_long_finetune | 187 | 26.36 | 0.836 | Full registration with longer finetune |
| Sequence 02 | partition_mid_finetune | 49 | 30.15 | 0.884 | Best photometric quality after partition selection |
| Sequence 03 | partition_best_finetune | 120 | 26.33 | 0.813 | Partition training improves weak baseline |
| Sequence 04 | partition_best_finetune | 80 | 28.36 | 0.811 | Best balance of rendering and geometry |
| Sequence 05 | full_finetune | 422 | 26.39 | 0.812 | Complete large sequence training |

Advanced methods have also been run on Sequence 04 for controlled comparison.

| Method | PSNR | SSIM | Status |
| --- | ---: | ---: | --- |
| Controlled 3DGS finetune | 28.36 | 0.811 | Main result |
| Mip Splatting official | 27.77 | 0.811 | Completed |
| 2DGS official | 27.18 | 0.796 | Completed rendering evaluation |
| Depth RegularizedGS | 15.92 | 0.662 | Completed but treated as failed ablation |

## Repository Layout

| Path | Content |
| --- | --- |
| report.pdf | Final technical report |
| report.tex | LaTeX source for the final report |
| beamer.pdf | Presentation PDF |
| beamer.tex | LaTeX source for the presentation |
| data | Shared indoor dataset extracted from data.zip |
| scripts | Data preparation, COLMAP, training, evaluation, PLY export, figure generation |
| tables | CSV tables used by the report and beamer |
| figures | Result figures and beamer image pairs |
| work | Selected final variants and advanced method outputs |
| external | Mip Splatting, 2DGS, Depth RegularizedGS source trees and local compatibility patches |
| docs | Reproduction notes and collaborator task notes |

Large binary files are managed with Git LFS. Old zip packages, COLMAP database files, repeated render folders, build products, and exploratory failed variants are intentionally ignored because they can be regenerated from the included data and scripts.

## Main Artifacts for Review

For grading or quick inspection, open these files first.

| File | Purpose |
| --- | --- |
| report.pdf | Complete written report with theory, method, experiments, failure analysis, and delivery guide |
| beamer.pdf | Presentation slides with clear image pairs and result tables |
| tables/render_quality_summary.csv | PSNR and SSIM for all final sequences |
| tables/geometry_summary.csv | Global geometry accuracy against reference point clouds |
| tables/local_accuracy_summary.csv | Local region accuracy statistics |
| tables/advanced_methods_summary.csv | Mip Splatting, 2DGS, Depth RegularizedGS comparison |
| figures/final_render_quality.png | Final rendering metric summary |
| figures/advanced_methods_comparison.png | Advanced method quantitative comparison |

## Reproduction Entry Points

The experiments were run with

```powershell
C:\Users\gjy20\.conda\envs\exp3\python.exe
```

For detailed reproduction commands, read

```text
docs/REPRODUCE.md
```

For suggested collaborator work, read

```text
docs/COLLABORATION.md
```

## Notes for Collaborators

The strongest immediate optimization targets are Sequence 01 low PSNR tail, Sequence 03 weak texture regions, Sequence 04 reflective wall and cabinet regions, and Sequence 05 large scale floating Gaussian artifacts. The report explains the observation that black high contrast lines are reconstructed more reliably than white reflective surfaces. This is consistent with the feature extraction and photometric optimization mechanisms used by COLMAP and 3DGS.
