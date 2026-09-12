# One-shot setup of the training PC (Windows + NVIDIA GPU).
# Installs Git, Python 3.12, VC++ runtime, enables SSH for the dev laptop,
# clones FlyMonster, installs CUDA PyTorch, downloads data and runs the GPU test.

$ErrorActionPreference = "Stop"

# Re-run as administrator if needed.
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -NoExit -File `"$PSCommandPath`""
    exit
}

$RepoUrl = "https://github.com/SuselMan/FlyMonster.git"
$Dir = "C:\projects\FlyMonster"
$LaptopKey = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMMjRrnmTpW+Im/vdfHOwmqlk1Uy8F1GM1ShTZfHQl7b flymonster-laptop"

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}

Step "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if ($LASTEXITCODE -ne 0) { throw "nvidia-smi not found: install the NVIDIA driver first" }

Step "Git, Python 3.12, VC++ runtime"
foreach ($id in "Git.Git", "Python.Python.3.12", "Microsoft.VCRedist.2015+.x64") {
    winget install --id $id -e --silent --accept-package-agreements --accept-source-agreements
}
Refresh-Path

Step "SSH server"
if ((Get-WindowsCapability -Online -Name OpenSSH.Server*).State -ne "Installed") {
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
}
Set-Service sshd -StartupType Automatic
Start-Service sshd
if (-not (Get-NetFirewallRule -Name sshd -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name sshd -DisplayName "OpenSSH Server" -Protocol TCP -LocalPort 22 -Action Allow -Direction Inbound | Out-Null
}
# Administrators read keys from ProgramData, not from the user profile.
$keys = "C:\ProgramData\ssh\administrators_authorized_keys"
if (-not (Test-Path $keys) -or -not (Select-String -Path $keys -SimpleMatch $LaptopKey -Quiet)) {
    Add-Content -Path $keys -Value $LaptopKey -Encoding ascii
}
icacls $keys /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F" | Out-Null
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
    -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -PropertyType String -Force | Out-Null

Step "Repository"
if (Test-Path "$Dir\.git") { git -C $Dir pull } else {
    New-Item -ItemType Directory -Force (Split-Path $Dir) | Out-Null
    git clone $RepoUrl $Dir
}
Set-Location $Dir

Step "Python environment (CUDA PyTorch)"
if (-not (Test-Path .venv)) { py -3.12 -m venv .venv }
.venv\Scripts\python -m pip install -q --upgrade pip
.venv\Scripts\pip install -q numpy pandas pyarrow
Write-Host "Downloading CUDA PyTorch (~2.5 GB)..."
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126 --progress-bar on
.venv\Scripts\python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.__version__, torch.cuda.get_device_name(0))"

Step "Data"
.venv\Scripts\python scripts\download_data.py

Step "GPU test: 1 s of fly life"
.venv\Scripts\python scripts\smoke_test.py --ms 1000 --device cuda
.venv\Scripts\python scripts\smoke_test.py --ms 1000 --device cuda --batch 32

Step "Done. Send these to the laptop:"
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.PrefixOrigin -in "Dhcp", "Manual" -and $_.IPAddress -notlike "169.*" }).IPAddress
Write-Host "  user: $env:USERNAME"
Write-Host "  ip:   $($ip -join ', ')"
