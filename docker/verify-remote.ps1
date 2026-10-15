param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$ApiKey = "",
    [string]$Image = "",
    [string]$Output = "auto-response.json"
)

# Verify readiness and optionally perform one real remote inference request.
$ErrorActionPreference = "Stop"
$normalizedBaseUrl = $BaseUrl.TrimEnd("/")

Write-Host "Checking $normalizedBaseUrl/health"
& curl.exe --fail-with-body --silent --show-error "$normalizedBaseUrl/health"
if ($LASTEXITCODE -ne 0) {
    throw "SAMPoly health check failed."
}
Write-Host ""

if ($Image) {
    $imagePath = [System.IO.Path]::GetFullPath($Image)
    if (-not (Test-Path -LiteralPath $imagePath -PathType Leaf)) {
        throw "Test image not found: $imagePath"
    }
    $arguments = @(
        "--fail-with-body", "--silent", "--show-error",
        "-X", "POST", "$normalizedBaseUrl/v1/auto/infer",
        "-F", "image=@$imagePath",
        "-o", ([System.IO.Path]::GetFullPath($Output))
    )
    if ($ApiKey) {
        $arguments += @("-H", "X-API-Key: $ApiKey")
    }
    & curl.exe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "SAMPoly remote inference request failed."
    }
    Write-Host "Inference response saved to: $([System.IO.Path]::GetFullPath($Output))"
}
