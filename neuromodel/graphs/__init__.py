from .graphs2d import line, lines
from .graphs3d import tuningcurve, regression
from .utils_bokeh import tweak_fig
from .graph import Graph

# remove LAPACK/scipy harmless warning (see https://github.com/scipy/scipy/issues/5998)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
