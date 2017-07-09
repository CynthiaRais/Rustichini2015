from .graphs2d import line, lines
from .utils_bokeh import tweak_fig
from .graph import Graph

from .graph import blue_pale, blue_dark

# remove LAPACK/scipy harmless warning (see https://github.com/scipy/scipy/issues/5998)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
