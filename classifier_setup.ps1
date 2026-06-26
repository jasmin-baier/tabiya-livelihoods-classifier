# See instructions: https://github.com/tabiya-tech/tabiya-livelihoods-classifier
Set-Location "C:\Users\jasmi\OneDrive - Nexus365\Documents\GitHub\tabiya-livelihoods-classifier"
Set-ExecutionPolicy Unrestricted -Scope CurrentUser
conda env update -f .\environment.yml --prune
conda activate tabiya-livelihoods-classifier