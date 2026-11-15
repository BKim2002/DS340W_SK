<#
.SYNOPSIS
Shows all eight retained results, with an optional live rank 8 comparison.
#>
[CmdletBinding()]
param([switch]$LiveRank8)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$comparisonPath = Join-Path $projectRoot 'data/results/scaling-comparison/comparison.json'
$comparison = Get-Content -LiteralPath $comparisonPath -Raw | ConvertFrom-Json
if ($comparison.verified -ne $true -or $comparison.aligned_test_rows -ne 436 -or
    $comparison.runs.Count -ne 4) {
    throw 'The verified four-rank comparison is missing or incomplete.'
}

$accuracyRows = @()
$resourceRows = @()
$referenceIds = $null
foreach ($pair in $comparison.runs) {
    foreach ($mode in @('original', 'modified')) {
        $run = $pair.$mode
        $manifestPath = Join-Path $projectRoot "data/manifests/retained-$($run.experiment_id).json"
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $correct = @($run.predictions | Where-Object { $_.label -eq $_.prediction }).Count
        $ids = ($run.predictions | ForEach-Object { "$($_.test_id):$($_.label)" }) -join ','
        if ($null -eq $referenceIds) { $referenceIds = $ids }
        $expectedMode = if ($mode -eq 'original') { 'original' } else { 'sqrt_rank' }
        if ($manifest.status -ne 'verified' -or $manifest.final_unseen_accessed -ne $false -or
            $manifest.config_sha256 -ne $run.config_sha256 -or
            $manifest.correct_count -ne $correct -or $run.correct_count -ne $correct -or
            $manifest.checkpoint_step -ne $run.checkpoint_step -or
            $run.predictions.Count -ne 436 -or $ids -ne $referenceIds -or
            $run.rank -ne $pair.rank -or $run.seed -ne 42 -or
            $run.alpha -ne (2 * $pair.rank) -or $run.scaling_mode -ne $expectedMode -or
            [Math]::Abs($run.accuracy_percent - (100 * $correct / 436)) -gt 0.000001) {
            throw "Inconsistent retained result: $($run.experiment_id)"
        }
    }
    if ($pair.original.trainable_parameters -ne $pair.modified.trainable_parameters) {
        throw "Trainable parameter counts differ at rank $($pair.rank)."
    }
    $accuracyRows += [pscustomobject]@{
        Rank = $pair.rank
        Alpha = $pair.original.alpha
        Original = '{0}/436 ({1:F2}%)' -f $pair.original.correct_count, $pair.original.accuracy_percent
        Modified = '{0}/436 ({1:F2}%)' -f $pair.modified.correct_count, $pair.modified.accuracy_percent
        'Change pp' = '{0:+0.00;-0.00;0.00}' -f ($pair.modified.accuracy_percent - $pair.original.accuracy_percent)
    }
    $resourceRows += [pscustomobject]@{
        Rank = $pair.rank
        'Orig min' = '{0:F1}' -f ($pair.original.runtime_seconds / 60)
        'Mod min' = '{0:F1}' -f ($pair.modified.runtime_seconds / 60)
        'Orig GiB' = '{0:F3}' -f $pair.original.peak_allocated_gib
        'Mod GiB' = '{0:F3}' -f $pair.modified.peak_allocated_gib
        Parameters = '{0:N0}' -f $pair.original.trainable_parameters
    }
}
if ((($accuracyRows | ForEach-Object { $_.Rank }) -join ',') -ne '2,4,8,16') {
    throw 'Expected ranks 2, 4, 8, and 16 in order.'
}

function Show-Comparison {
    Write-Host "`nRETAINED TRAINING RESULTS - original and modified scaling"
    Write-Host 'Same 436 course Test examples; seed 42; alpha = 2 x rank.'
    Write-Host 'Original: alpha/rank. Modified: alpha/sqrt(rank).'
    Write-Host ($accuracyRows | Format-Table -AutoSize | Out-String -Width 120)
    Write-Host 'Recorded process time, peak allocated GPU memory, and trainable parameters'
    Write-Host ($resourceRows | Format-Table -AutoSize | Out-String -Width 120)
    Write-Host 'Rank 8 improved by four correct answers; ranks 2, 4, and 16 declined.'
    Write-Host 'These are selected-checkpoint results from one training seed.'
    Write-Host 'Course Test also selected checkpoints. Final Unseen remains reserved.'
    Write-Host 'This table reads saved records; it does not execute the models.'
}

Show-Comparison
if ($LiveRank8) {
    $wslRoot = (& wsl.exe -d Ubuntu-22.04 -- wslpath -a $projectRoot).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($wslRoot)) {
        throw 'Could not resolve the project folder in Ubuntu-22.04.'
    }
    Write-Host "`nLIVE EVALUATION - ORIGINAL rank 8"
    & wsl.exe -d Ubuntu-22.04 --cd $wslRoot -- /home/tjrdu/.venvs/ds340w/bin/python scripts/demo_baseline_eval.py
    if ($LASTEXITCODE -ne 0) { throw 'Original rank 8 live evaluation failed.' }
    Write-Host "`nLIVE EVALUATION - MODIFIED rank 8"
    & wsl.exe -d Ubuntu-22.04 --cd $wslRoot -- /home/tjrdu/.venvs/ds340w/bin/python scripts/demo_scaling_eval.py --rank 8
    if ($LASTEXITCODE -ne 0) { throw 'Modified rank 8 live evaluation failed.' }
    Write-Host "`nBoth live evaluations passed and matched their retained predictions."
    Show-Comparison
}
