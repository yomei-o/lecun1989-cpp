# Downloads the raw MNIST IDX files into .\data (Windows PowerShell friendly).
# Uses only .NET built-ins for the gzip step -- no external tools required.
#
#   powershell -ExecutionPolicy Bypass -File get_mnist.ps1
#
$ErrorActionPreference = 'Stop'
$base = 'https://storage.googleapis.com/cvdf-datasets/mnist'
$files = @(
    'train-images-idx3-ubyte',
    'train-labels-idx1-ubyte',
    't10k-images-idx3-ubyte',
    't10k-labels-idx1-ubyte'
)
$dir = Join-Path $PSScriptRoot 'data'
New-Item -ItemType Directory -Force -Path $dir | Out-Null

foreach ($f in $files) {
    $out = Join-Path $dir $f
    if (Test-Path $out) { Write-Host "have  $f"; continue }
    $gz = "$out.gz"
    Write-Host "fetch $f.gz ..."
    Invoke-WebRequest -Uri "$base/$f.gz" -OutFile $gz
    # gunzip via .NET
    $in  = [System.IO.File]::OpenRead($gz)
    $gzs = New-Object System.IO.Compression.GzipStream($in, [System.IO.Compression.CompressionMode]::Decompress)
    $outfs = [System.IO.File]::Create($out)
    $gzs.CopyTo($outfs)
    $outfs.Close(); $gzs.Close(); $in.Close()
    Remove-Item $gz
    Write-Host "  -> $f ($((Get-Item $out).Length) bytes)"
}
Write-Host "MNIST ready in $dir"
