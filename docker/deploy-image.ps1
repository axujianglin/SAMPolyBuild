param(
    [Parameter(Mandatory = $true)]
    [string]$Archive,
    [string]$EnvFile = ".env"
)

# Load an offline image archive and start the remote-access Compose service.
$ErrorActionPreference = "Stop"
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$archivePath = [System.IO.Path]::GetFullPath($Archive)
$envPath = [System.IO.Path]::GetFullPath($EnvFile)
$composePath = Join-Path $scriptDirectory "compose.deploy.yaml"

if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw "Image archive not found: $archivePath"
}
if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
    throw "Environment file not found: $envPath"
}

docker load --input $archivePath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to import Docker image archive: $archivePath"
}

docker compose --env-file $envPath -f $composePath up -d
if ($LASTEXITCODE -ne 0) {
    throw "Failed to start the SAMPoly deployment service."
}

docker compose --env-file $envPath -f $composePath ps
