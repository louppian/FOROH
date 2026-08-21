param(
    [string]$Target = "D:\FOROH",
    [string]$Branch = "reproduce-paper"
)

$ErrorActionPreference = "Stop"
$Repo = "https://github.com/louppian/FOROH.git"

Write-Host "FOROH handoff setup"
Write-Host "Target : $Target"
Write-Host "Branch : $Branch"

$parent = Split-Path -Parent $Target
if (-not (Test-Path $parent)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
}

if (-not (Test-Path $Target)) {
    Write-Host "Cloning repository..."
    git clone --branch $Branch --single-branch $Repo $Target
} elseif (Test-Path (Join-Path $Target ".git")) {
    Write-Host "Existing repository found. Updating..."
    Push-Location $Target
    try {
        git fetch origin
        git switch $Branch
        git pull origin $Branch
    } finally {
        Pop-Location
    }
} else {
    throw "$Target exists but is not a Git repository. Move/remove it first or use another -Target."
}

Push-Location $Target
try {
    Write-Host ""
    Write-Host "Repository status"
    git status --short --branch

    $required = @(
        "HANDOFF.md",
        "REPOSITORY_STRUCTURE.md",
        "Experiment\PLAN.md",
        "Experiment\00_Inventory"
    )

    Write-Host ""
    Write-Host "Handoff file check"
    $failed = $false
    foreach ($p in $required) {
        if (Test-Path $p) {
            Write-Host "[OK] $p"
        } else {
            Write-Host "[MISSING] $p"
            $failed = $true
        }
    }

    Write-Host ""
    Write-Host "Local-only assets that Git cannot guarantee:"
    foreach ($p in @("data", "outputs", "Result")) {
        if (Test-Path $p) {
            $count = (Get-ChildItem $p -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
            Write-Host "[FOUND] $p ($count files)"
        } else {
            Write-Host "[NOT FOUND] $p"
        }
    }

    $weights = Get-ChildItem -Recurse -File -Include *.pt,*.pth,*.ckpt -ErrorAction SilentlyContinue
    Write-Host "Checkpoint files visible locally: $($weights.Count)"

    if ($failed) {
        throw "Required handoff files are missing. Check branch/update status."
    }

    Write-Host ""
    Write-Host "Setup complete: $Target"
    Write-Host "Read HANDOFF.md first."
} finally {
    Pop-Location
}
