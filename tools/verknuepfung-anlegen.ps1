<#
.SYNOPSIS
    Legt die Desktop-Verknuepfung "KI-Dashboard" an.

.DESCRIPTION
    Die Verknuepfung startet den Server und oeffnet den Browser - beides macht
    dashboard.py selbst, die Verknuepfung ruft nur das venv-Python damit auf.

    Bewusst ohne .bat-Zwischendatei: eine Verknuepfung direkt auf pythonw.exe
    startet ohne Konsolenfenster. Fuer den Fall, dass etwas schiefgeht, gibt es
    zusaetzlich eine zweite Verknuepfung mit sichtbarer Konsole.

    Rueckbau: beide Verknuepfungen vom Desktop loeschen. Mehr wird nicht angelegt.

.EXAMPLE
    .\tools\verknuepfung-anlegen.ps1
    .\tools\verknuepfung-anlegen.ps1 -Entfernen
#>
[CmdletBinding()]
param([switch]$Entfernen)

$ErrorActionPreference = 'Stop'

$RepoDir  = Split-Path $PSScriptRoot -Parent
$Desktop  = [Environment]::GetFolderPath('Desktop')
$Pythonw  = Join-Path $RepoDir '.venv\Scripts\pythonw.exe'
$Python   = Join-Path $RepoDir '.venv\Scripts\python.exe'
$Skript   = Join-Path $RepoDir 'dashboard.py'

$Ziele = @(
    @{ Name = 'KI-Dashboard.lnk';         Exe = $Pythonw; Beschreibung = 'Live-Dashboard fuer lokale LLM-Inferenz' }
    @{ Name = 'KI-Dashboard (Konsole).lnk'; Exe = $Python;  Beschreibung = 'Live-Dashboard mit sichtbarer Konsole - fuer die Fehlersuche' }
)

if ($Entfernen) {
    foreach ($z in $Ziele) {
        $pfad = Join-Path $Desktop $z.Name
        if (Test-Path $pfad) { Remove-Item -LiteralPath $pfad -Force; Write-Host "Entfernt: $($z.Name)" }
    }
    return
}

foreach ($pfad in @($Pythonw, $Python, $Skript)) {
    if (-not (Test-Path $pfad)) { throw "Fehlt: $pfad  (venv angelegt?)" }
}

# Ein Symbol, das nicht nach "irgendein Python-Skript" aussieht.
$Icon = "$env:SystemRoot\System32\shell32.dll,13"

$shell = New-Object -ComObject WScript.Shell
foreach ($z in $Ziele) {
    $pfad = Join-Path $Desktop $z.Name
    $lnk = $shell.CreateShortcut($pfad)
    $lnk.TargetPath       = $z.Exe
    $lnk.Arguments        = "`"$Skript`""
    $lnk.WorkingDirectory = $RepoDir      # sonst findet Python das Paket kibench nicht
    $lnk.Description      = $z.Beschreibung
    $lnk.IconLocation     = $Icon
    $lnk.Save()
    Write-Host "Angelegt: $pfad"
}

Write-Host ''
Write-Host 'Doppelklick startet Sampler und Live-Ansicht und oeffnet den Browser.' -ForegroundColor Green
Write-Host 'Beenden: das Konsolenfenster schliessen bzw. den python-Prozess beenden.'
