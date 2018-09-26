# Figure Comparator

Download the pdf of the paper and place it in this directory under the name `Rustichini2015.pdf`
```
wget https://www.physiology.org/doi/pdf/10.1152/jn.00184.2015 -O Rustichini2015.pdf
```

Run the `extract_original.py` script from the root of this directory. It will extract the individual figures from the pdf:
```
python extract_originals.py
```

After having generated the figure from the python code in the `../figures` directory, run the `prepare_overlays.py` script. It will create version of the figures that are translated and scaled to be superposable to the one of the article.
```
python prepare_overlays.py
```

Then, open the `figures.html` or `figure_replicated.html` file in any web browser.
