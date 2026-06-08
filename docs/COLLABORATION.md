# Collaboration Notes

This file records the strongest next steps for collaborators. The current repository already contains full report materials, final tables, generated figures, selected final outputs, and advanced method ablations.

## Current Baseline

The strongest controlled result is Sequence 02 with PSNR 30.15 and SSIM 0.884. Sequence 04 reaches PSNR 28.36 and is the best candidate for further advanced method work because it has complete controlled 3DGS, Mip Splatting, 2DGS, and Depth RegularizedGS outputs.

## Immediate Optimization Targets

Sequence 01

The low PSNR tail is caused by weakly observed views and reflective regions. The next useful step is a stricter frame filter and a longer finetune using a lower learning rate.

Sequence 03

Weak texture and repeated structure reduce stability. The next useful step is partition training with overlap and line or plane regularization.

Sequence 04

This sequence is the main advanced method comparison target. The current controlled 3DGS result is stronger than the tested official Mip Splatting and 2DGS settings. Further work should tune Mip Splatting with a longer schedule, test antialiasing settings, and run a cleaner train test split.

Sequence 05

This is the largest sequence and has the most complete camera coverage. The main problem is large scale floater accumulation. The next useful step is spatial subdivision and post training opacity pruning.

## Method Tasks

Data side

Improve frame selection by using blur score, exposure score, and pairwise image overlap. Keep high contrast wall lines, cabinet edges, carpet boundaries, and door frames because they support COLMAP and 3DGS optimization.

COLMAP side

Compare exhaustive and sequential matching on Sequence 02 and Sequence 04. Track registration rate, sparse point count, mean reprojection error, and final PSNR together. Sequence 02 currently has only 49 registered images after partition selection, so additional matching settings may recover more views.

3DGS side

Tune densify until, densify interval, densify fraction, maximum new Gaussians per densification step, SSIM loss, edge loss, opacity regularization, and finetune learning rate. The current custom script already exposes these knobs.

Advanced methods

Mip Splatting and 2DGS are already installed and evaluated. Depth RegularizedGS runs to completion but currently performs poorly with sparse depth fallback. A collaborator can replace sparse depth fallback with dense monocular depth maps if ZoeDepth or another depth model is configured cleanly.

Geometry

Current global point cloud distance is stricter than local measurement. Local median errors for Sequence 04 and Sequence 05 are below 15 cm, and the best local median errors are below 10 cm. Future work should add explicit wall plane and cabinet plane measurements to make the accuracy evaluation more interpretable.

## Failure Observation

White reflective cabinet surfaces and walls are harder to recover because they contain weak gradients and view dependent highlights. Black lines, door frames, wires, printed text, and carpet boundaries are reconstructed more reliably because they provide high contrast, stable local gradients and repeated feature support. Useful future directions are reflective masks, line supervision, plane constraints, depth regularization with dense depth, Mip Splatting antialiasing, and spatial partition training.
