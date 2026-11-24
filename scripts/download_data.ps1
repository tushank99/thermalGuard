# Downloads NASA C-MAPSS (public, NASA Prognostics Data Repository mirror) into data/raw/cmapss
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force data/raw | Out-Null
Invoke-WebRequest -Uri "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip" -OutFile data/raw/outer.zip
Expand-Archive data/raw/outer.zip -DestinationPath data/raw/outer -Force
Expand-Archive "data/raw/outer/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip" -DestinationPath data/raw/cmapss -Force
Remove-Item -Recurse -Force data/raw/outer, data/raw/outer.zip
Write-Host "C-MAPSS ready in data/raw/cmapss"
