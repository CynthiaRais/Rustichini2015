# Rustichini2015
Reproduction of "A neuro-computational model of economic decisions"


## Install

Requires Python 3.5 or higher. Tested on Linux and macOS.
```
pip install -r requirements.txt
```

To produce and manipulate figures, you will also need to install PhantomJS:
```
npm install -g phantomjs
```
And Inkscape and imagemagick. In macOS, with [Homebrew](https://brew.sh/):
```
brew cast install inkscape
brew install imagemagick
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
