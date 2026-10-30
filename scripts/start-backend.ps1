param([int]$Port=8000)
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH='backend'
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port $Port
