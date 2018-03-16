# Rustichini2015
Reproduction of "a neuro-computational model of economic decisions"


## Install

Requires Python 3.3 or higher.
```
pip install -r requirements.txt
```

You also need to install PhantomJS:
```
npm install -g phantomjs
```


## Generating graphs

In the `notebooks/` folder, you can run:
```
python fig4_data.py
python fig5_data.py
python fig6_data.py
```
This will precompute the data necessary to compute the graphs. Alternatively,
you can run the notebooks and the data will be computed as necessary.
