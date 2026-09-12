@echo off
rem Double-click on the training PC. Uses the .ps1 next to it, or downloads it from GitHub.
set PS1=%~dp0gpu-pc-setup.ps1
if not exist "%PS1%" (
  set PS1=%TEMP%\gpu-pc-setup.ps1
  powershell -NoProfile -Command "Invoke-WebRequest https://raw.githubusercontent.com/SuselMan/FlyMonster/main/setup/gpu-pc-setup.ps1 -OutFile $env:TEMP\gpu-pc-setup.ps1"
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%"
