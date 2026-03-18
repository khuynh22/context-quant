# schedule_task.ps1
# Registers a Windows Task Scheduler job that runs retrain_daily.py
# every weekday (Mon-Fri) at 6:00 AM, before US market open.
#
# Run ONCE as Administrator:
#   powershell -ExecutionPolicy Bypass -File schedule_task.ps1

$TaskName   = "ContextQuant-DailyRetrain"
$RepoRoot   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$UvExe      = Join-Path $RepoRoot ".venv\Scripts\uv.exe"
$Script     = Join-Path $RepoRoot "retrain_daily.py"
$LogDir     = Join-Path $RepoRoot "logs"

# Fallback: use uv from PATH if not found in venv
if (-not (Test-Path $UvExe)) {
    $UvExe = (Get-Command uv -ErrorAction Stop).Source
}

# Create logs directory if it doesn't exist
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$Action = New-ScheduledTaskAction `
    -Execute $UvExe `
    -Argument "run python `"$Script`"" `
    -WorkingDirectory $RepoRoot

# Run Mon–Fri at 6:00 AM
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "06:00AM"

$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -StartWhenAvailable `          # run missed job on next startup if PC was off
    -RunOnlyIfNetworkAvailable

$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

# Remove old task if it exists
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed old task '$TaskName'."
}

Register-ScheduledTask `
    -TaskName  $TaskName `
    -Action    $Action `
    -Trigger   $Trigger `
    -Settings  $Settings `
    -Principal $Principal `
    -Description "Retrain ContextQuantFusionNet daily on all tickers in tickers.txt"

Write-Host ""
Write-Host "Scheduled task '$TaskName' registered successfully." -ForegroundColor Green
Write-Host "Runs every weekday at 6:00 AM using: $UvExe"
Write-Host "Logs written to: $LogDir\retrain.log"

# ── Task 2: Paper trading session at 9:30 AM (market open) ───────────────────
$PaperTaskName = "ContextQuant-PaperTrading"
$PaperScript   = Join-Path $RepoRoot "paper_trader.py"

$PaperAction = New-ScheduledTaskAction `
    -Execute $UvExe `
    -Argument "run python `"$PaperScript`"" `
    -WorkingDirectory $RepoRoot

# Run Mon–Fri at 9:30 AM (US market open)
$PaperTrigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "09:30AM"

$PaperSettings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

if (Get-ScheduledTask -TaskName $PaperTaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $PaperTaskName -Confirm:$false
    Write-Host "Removed old task '$PaperTaskName'."
}

Register-ScheduledTask `
    -TaskName  $PaperTaskName `
    -Action    $PaperAction `
    -Trigger   $PaperTrigger `
    -Settings  $PaperSettings `
    -Principal $Principal `
    -Description "Run ContextQuant paper trading session at market open"

Write-Host ""
Write-Host "Scheduled task '$PaperTaskName' registered successfully." -ForegroundColor Green
Write-Host "Runs every weekday at 9:30 AM (market open)"
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  Run retrain now:      Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Run paper trade now:  Start-ScheduledTask -TaskName '$PaperTaskName'"
Write-Host "  Check retrain log:    Get-Content '$LogDir\retrain.log' -Tail 40"
Write-Host "  View signals today:   Get-Content (Get-ChildItem paper_trading\journal.csv)"
Write-Host "  Performance report:   uv run python paper_trader.py --report"
Write-Host "  Remove all tasks:     Unregister-ScheduledTask -TaskName '$TaskName','$PaperTaskName' -Confirm:`$false"
