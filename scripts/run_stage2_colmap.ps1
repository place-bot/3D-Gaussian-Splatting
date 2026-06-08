param(
    [string]$Sequence = "Sequence_01",
    [string]$Variant = "improved",
    [ValidateSet("sequential", "exhaustive")]
    [string]$Matcher = "exhaustive",
    [ValidateSet("standard", "robust")]
    [string]$Profile = "robust"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Colmap = Join-Path $Root "..\stage1\tools\COLMAP.bat"
$WorkDir = Join-Path $Root "work\$Sequence\$Variant"
$ImageDir = Join-Path $WorkDir "input"
$ColmapDir = Join-Path $WorkDir "colmap"
$Database = Join-Path $ColmapDir "database.db"
$SparseDir = Join-Path $ColmapDir "sparse"
$Undistorted = Join-Path $WorkDir "undistorted"
$SparseText = Join-Path $Undistorted "sparse_text"
$Log = Join-Path $WorkDir "colmap_$Matcher.log"

if (-not (Test-Path -LiteralPath $ImageDir)) {
    throw "Image directory does not exist: $ImageDir"
}
if (-not (Test-Path -LiteralPath $Colmap)) {
    throw "COLMAP executable does not exist: $Colmap"
}

New-Item -ItemType Directory -Path $ColmapDir, $SparseDir -Force | Out-Null
if (Test-Path -LiteralPath $Database) {
    Remove-Item -LiteralPath $Database -Force
}
if (Test-Path -LiteralPath $SparseDir) {
    Remove-Item -LiteralPath $SparseDir -Recurse -Force
    New-Item -ItemType Directory -Path $SparseDir | Out-Null
}
if (Test-Path -LiteralPath $Undistorted) {
    Remove-Item -LiteralPath $Undistorted -Recurse -Force
}
if (Test-Path -LiteralPath $Log) {
    Remove-Item -LiteralPath $Log -Force
}

function Invoke-Colmap {
    param([string[]]$Arguments)
    Add-Content -LiteralPath $Log -Value ("`nCOLMAP " + ($Arguments -join " "))
    $commandParts = @("`"$Colmap`"") + ($Arguments | ForEach-Object {
        $escaped = $_ -replace '"', '\"'
        "`"$escaped`""
    })
    $commandLine = ($commandParts -join " ") + " >> `"$Log`" 2>&1"
    cmd.exe /c $commandLine
    $exitCode = $LASTEXITCODE
    if (Test-Path -LiteralPath $Log) {
        Get-Content -LiteralPath $Log -Tail 20
    }
    if ($exitCode -ne 0) {
        throw "COLMAP failed with exit code $exitCode"
    }
}

Invoke-Colmap @(
    "feature_extractor",
    "--database_path", $Database,
    "--image_path", $ImageDir,
    "--ImageReader.single_camera", "1",
    "--ImageReader.camera_model", "SIMPLE_RADIAL",
    "--FeatureExtraction.use_gpu", "1",
    "--SiftExtraction.max_num_features", $(if ($Profile -eq "robust") { "16384" } else { "8192" }),
    "--SiftExtraction.peak_threshold", $(if ($Profile -eq "robust") { "0.003" } else { "0.00667" }),
    "--SiftExtraction.edge_threshold", $(if ($Profile -eq "robust") { "20" } else { "10" }),
    "--SiftExtraction.estimate_affine_shape", $(if ($Profile -eq "robust") { "1" } else { "0" }),
    "--SiftExtraction.domain_size_pooling", $(if ($Profile -eq "robust") { "1" } else { "0" })
)

if ($Matcher -eq "sequential") {
    Invoke-Colmap @(
        "sequential_matcher",
        "--database_path", $Database,
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.guided_matching", $(if ($Profile -eq "robust") { "1" } else { "0" }),
        "--SequentialMatching.overlap", $(if ($Profile -eq "robust") { "40" } else { "20" }),
        "--TwoViewGeometry.min_num_inliers", $(if ($Profile -eq "robust") { "10" } else { "15" }),
        "--TwoViewGeometry.min_inlier_ratio", $(if ($Profile -eq "robust") { "0.15" } else { "0.25" })
    )
} else {
    Invoke-Colmap @(
        "exhaustive_matcher",
        "--database_path", $Database,
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.guided_matching", $(if ($Profile -eq "robust") { "1" } else { "0" }),
        "--TwoViewGeometry.min_num_inliers", $(if ($Profile -eq "robust") { "10" } else { "15" }),
        "--TwoViewGeometry.min_inlier_ratio", $(if ($Profile -eq "robust") { "0.15" } else { "0.25" })
    )
}

Invoke-Colmap @(
    "mapper",
    "--database_path", $Database,
    "--image_path", $ImageDir,
    "--output_path", $SparseDir,
    "--Mapper.ba_global_function_tolerance", "0.000001",
    "--Mapper.init_min_num_inliers", $(if ($Profile -eq "robust") { "30" } else { "100" }),
    "--Mapper.init_min_tri_angle", $(if ($Profile -eq "robust") { "4" } else { "16" }),
    "--Mapper.abs_pose_min_num_inliers", $(if ($Profile -eq "robust") { "12" } else { "30" }),
    "--Mapper.abs_pose_min_inlier_ratio", $(if ($Profile -eq "robust") { "0.10" } else { "0.25" }),
    "--Mapper.ba_refine_principal_point", $(if ($Profile -eq "robust") { "1" } else { "0" })
)

$ModelDirs = Get-ChildItem -LiteralPath $SparseDir -Directory | Sort-Object {
    $imagesBin = Join-Path $_.FullName "images.bin"
    if (Test-Path -LiteralPath $imagesBin) {
        (Get-Item -LiteralPath $imagesBin).Length
    } else {
        0
    }
} -Descending
if ($ModelDirs.Count -eq 0) {
    throw "COLMAP mapper did not create any sparse model"
}
$ModelDir = $ModelDirs[0].FullName
Write-Host "Using sparse model $($ModelDirs[0].Name)"

Invoke-Colmap @(
    "image_undistorter",
    "--image_path", $ImageDir,
    "--input_path", $ModelDir,
    "--output_path", $Undistorted,
    "--output_type", "COLMAP"
)

New-Item -ItemType Directory -Path $SparseText | Out-Null
Invoke-Colmap @(
    "model_converter",
    "--input_path", (Join-Path $Undistorted "sparse"),
    "--output_path", $SparseText,
    "--output_type", "TXT"
)

Write-Host "COLMAP completed for $Sequence $Variant"
