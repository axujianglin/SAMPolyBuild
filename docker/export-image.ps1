param(
    [string]$Image = "sampoly-auto-api:cu117",
    [string]$Output = "sampoly-auto-api-cu117.tar"
)

# Export a fully built image for an offline Windows or Linux deployment host.
$ErrorActionPreference = "Stop"
$outputPath = [System.IO.Path]::GetFullPath($Output)
$outputDirectory = Split-Path -Parent $outputPath

docker image inspect $Image | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker image '$Image' does not exist. Build it before exporting."
}

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
docker save --output $outputPath $Image
if ($LASTEXITCODE -ne 0) {
    throw "Failed to export Docker image '$Image'."
}

Write-Host "Image exported to: $outputPath"
