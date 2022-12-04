![](https://img.shields.io/badge/status-finished-green?style=flat-square)
![](https://img.shields.io/badge/Python-blue?style=flat-square&logo=python&color=blue&logoColor=white&labelColor=gray)
![](https://img.shields.io/badge/dependency-numpy-blue?style=flat-square)

# ACA-SPSD

In this repository we provide the implementation of the project "Volume maximization for cross approximation" of the class Low-Rank Approximation Techiques, MATH-403  <br/>
_Authors: Fabio Matti, Anna Paulish_

## Instructions

You can reproduce our results with
```[bash]
git clone https://github.com/FMatti/ACA-SPSD.git
cd ACA-SPSD
python main.py
```

This takes about 5 minutes. If dependency problems arise, you can image our Python environment using

```[bash]
python -m venv .venv

source .venv/bin/activate   # on Linux, macOS
.venv\Scripts\activate.bat  # on Windows

pip install --upgrade pip
pip install -r requirements.txt
```

Our implementations require a Python version $\geq$ 3.8.

## File structure
Our implementations are located in the `src/` directory. Our results can be found in the Jupyter notebook `main.ipynb` or equivalently reproduced by running the Python script `main.py`.

```
ACA-SPSD
│   README.md
|   main.ipynb             (Jupyter notebook with our results)
|   main.py                (equivalent Python script with our results)
|
└───src
|   |   algorithms.py      (implementations of the two algorithms)
|   |   helpers.py         (helper functions)
|   |   matrices.py        (definition of the example matrices)
```
