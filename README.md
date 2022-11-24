# ACA-SPSD

## Instructions

Clone the repository
```[bash]
git clone https://github.com/FMatti/ACA-SPSD.git
cd ACA-SPSD
```

Create and activate a virtual environment
```[bash]
python -m venv .venv

source .venv/bin/activate (on Linux, macOS)
.venv\Scripts\activate.bat (on Windows)
```

Install the required packages
```[bash]
pip update --install pip
pip install -r requirements.txt
```

Reproduce our results (takes ~5 min)
```[bash]
python main.py
```