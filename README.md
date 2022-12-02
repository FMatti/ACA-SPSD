# ACA-SPSD

In this repository we provide the implementation of the project "Volume maximization for cross approximation" of the class Low-Rank Approximation Techiques, MATH-403  <br/>
_Authors: Fabio Matti, Anna Paulish_

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
pip install --upgrade pip
pip install -r requirements.txt
```

Reproduce our results (takes ~5 min)
```[bash]
python main.py
```
