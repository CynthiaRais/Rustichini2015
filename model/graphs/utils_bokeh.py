from bokeh import io
from bokeh import plotting
from bokeh.models import FixedTicker, AdaptiveTicker, FuncTickFormatter, LabelSet, ColumnDataSource
from bokeh.layouts import row, column, gridplot

COLOR_A = '#fa6900'
COLOR_B = '#69d2e7'

COLORS_AB       = [COLOR_A, COLOR_B]

plotting.output_notebook(hide_banner=True)

def tweak_fig(fig):
    tight_layout(fig)
    disable_minor_ticks(fig)
    disable_grid(fig)
    fig.toolbar.logo = None

def tight_layout(fig):
    fig.min_border_top    = 35
    fig.min_border_bottom = 35
    fig.min_border_right  = 35
    fig.min_border_left   = 35

def disable_minor_ticks(fig):
    #fig.axis.major_label_text_font_size = value('8pt')
    fig.axis.minor_tick_line_color = None
    fig.axis.major_tick_in = 0

def disable_grid(fig):
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None


def figure(*args, **kwargs):
    fig = plotting.figure(*args, **kwargs)
    tweak_fig(fig)
    return fig


    ## Removing returns

def show(*args, **kwargs):
    return plotting.show(*args, **kwargs)

def interact(*args, **kwargs):
    ipywidgets.widgets.interact(*args, **kwargs)

def select(name, options):
    return SelectionSlider(description=name,  options=list(options))
