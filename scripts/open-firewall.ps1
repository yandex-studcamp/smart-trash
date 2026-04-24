param(
    [int]$Port = 8000,
    [string]$RuleName = "Smart Trash Server"
)

$ErrorActionPreference = "Stop"
$currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
$isAdministrator = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdministrator) {
    throw "Run this script from an elevated PowerShell session."
}

$resolvedRuleName = "$RuleName ($Port)"
netsh advfirewall firewall add rule name="$resolvedRuleName" dir=in action=allow protocol=TCP localport=$Port
