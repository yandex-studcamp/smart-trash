param(
    [string]$ListenHost = "0.0.0.0",
    [int]$Port = 8000,
    [switch]$Reload,
    [string]$CorsAllowOrigins = "*"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

$env:APP_HOST = $ListenHost
$env:APP_PORT = "$Port"
$env:APP_RELOAD = if ($Reload.IsPresent) { "true" } else { "false" }
$env:APP_CORS_ALLOW_ORIGINS = $CorsAllowOrigins

Write-Host "Smart Trash Server"
Write-Host "Host: $ListenHost"
Write-Host "Port: $Port"

if ($ListenHost -eq "0.0.0.0") {
    $addresses = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
            $_.IPAddress -ne "127.0.0.1" -and
            $_.IPAddress -notlike "169.254*" -and
            $_.IPAddress -notlike "0.*"
        } |
        Select-Object -ExpandProperty IPAddress -Unique

    foreach ($address in $addresses) {
        Write-Host "LAN URL: http://$address`:$Port"
    }
}

$uvicornArgs = @(
    "run",
    "uvicorn",
    "app.main:app",
    "--app-dir",
    "src",
    "--host",
    $ListenHost,
    "--port",
    "$Port"
)

if ($Reload.IsPresent) {
    $uvicornArgs += "--reload"
}

uv @uvicornArgs
