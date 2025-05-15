# See instructions: https://github.com/tabiya-tech/tabiya-livelihoods-classifier
Set-Location "C:\Users\jasmi\Documents\GitHub\tabiya-livelihoods-classifier"
Set-ExecutionPolicy Unrestricted -Scope CurrentUser
venv\Scripts\Activate
poetry install
python -c "import nltk; nltk.download('punkt')"