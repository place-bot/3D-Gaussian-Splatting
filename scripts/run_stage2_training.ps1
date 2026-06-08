param(
    [string]$Sequence = "Sequence_01",
    [string]$Variant = "improved",
    [string]$SourceVariant = "",
    [string]$OutputVariant = "",
    [int]$Iterations = 9000,
    [int]$MaxSide = 960,
    [int]$DensifyUntil = 3500,
    [int]$DensifyEvery = 500,
    [double]$DensifyFraction = 0.08,
    [int]$DensifyMaxNew = 1600,
    [double]$LambdaSsim = 0.0,
    [double]$LambdaEdge = 0.0,
    [double]$LambdaOpacity = 0.0,
    [string]$ResumeCheckpoint = "",
    [double]$LrScale = 1.0,
    [int]$SaveRenderCount = 16
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = "C:\Users\gjy20\.conda\envs\exp3\python.exe"
$VsDevCmd = "C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat"
$Stage1Root = Join-Path $Root "..\stage1"
$CudaHome = Join-Path $Stage1Root "tools\cuda-12.1-local"
if ($SourceVariant -eq "") {
    $SourceVariant = $Variant
}
if ($OutputVariant -eq "") {
    $OutputVariant = $Variant
}
$SceneDir = Join-Path $Root "work\$Sequence\$SourceVariant\undistorted"
$OutputDir = Join-Path $Root "work\$Sequence\$OutputVariant\output"
$FiguresDir = Join-Path $Root "work\$Sequence\$OutputVariant\figures"
$TrainScript = Join-Path $Root "scripts\train_stage2_gsplat.py"
$LogPath = Join-Path $Root "work\$Sequence\$OutputVariant\training.log"

if (-not (Test-Path -LiteralPath $SceneDir)) {
    throw "Scene directory does not exist: $SceneDir"
}
if (-not (Test-Path -LiteralPath $TrainScript)) {
    throw "Training script does not exist: $TrainScript"
}

New-Item -ItemType Directory -Path $OutputDir, $FiguresDir | Out-Null

$ResumeArgs = ""
if ($ResumeCheckpoint -ne "") {
    if (-not (Test-Path -LiteralPath $ResumeCheckpoint)) {
        throw "Resume checkpoint does not exist: $ResumeCheckpoint"
    }
    $ResumeArgs = "--resume-checkpoint `"$ResumeCheckpoint`""
}

$Commands = @(
    "call `"$VsDevCmd`" -arch=x64 -vcvars_ver=14.44",
    "set `"CUDA_HOME=$CudaHome`"",
    "set `"CUDA_PATH=$CudaHome`"",
    "set `"CUDACXX=$CudaHome\bin\nvcc.exe`"",
    "set `"NVCC_PREPEND_FLAGS=-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`"",
    "set `"TORCH_CUDA_ARCH_LIST=8.9`"",
    "set `"MAX_JOBS=4`"",
    "set `"PATH=C:\Users\gjy20\.conda\envs\exp3;C:\Users\gjy20\.conda\envs\exp3\Scripts;$CudaHome\bin;$CudaHome\lib\x64;!PATH!`"",
    "`"$Python`" `"$TrainScript`" --scene-dir `"$SceneDir`" --output-dir `"$OutputDir`" --figures-dir `"$FiguresDir`" --iterations $Iterations --max-side $MaxSide --save-render-count $SaveRenderCount --densify-until $DensifyUntil --densify-every $DensifyEvery --densify-fraction $DensifyFraction --densify-max-new $DensifyMaxNew --lambda-ssim $LambdaSsim --lambda-edge $LambdaEdge --lambda-opacity $LambdaOpacity --lr-scale $LrScale $ResumeArgs > `"$LogPath`" 2>&1"
)

cmd.exe /v:on /c ($Commands -join " && ")
