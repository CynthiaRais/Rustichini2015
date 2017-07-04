from .utils_bokeh import figure, plotting, COLORS_AB


def line(xs, ys, std=None, title='', width=400, height=200, dots=True, legend=None, fig=None, color='#00a0b0', alpha=1.0, line_width=1, show=True, **kwargs):
    if fig is None:
        fig = figure(plot_width=width, plot_height=height, tools="save", title=title, **kwargs)

    fig.line(xs, ys, line_color=color, line_alpha=alpha, line_width=line_width, legend=legend, )
    if dots:
        fig.scatter(xs, ys, line_color=None, fill_color=color, size=4)
    if std is not None:
        assert len(std) == len(ys)
        x_std = list(xs) + list(reversed(xs))
        y_std = ([m_i + std_i for m_i, std_i in zip(ys, std)]
                 + list(reversed([m_i - std_i for m_i, std_i in zip(ys, std)])))
        fig.patch(x_std,  y_std, fill_color=color, fill_alpha=0.10, line_color=None)

    if show:
        plotting.show(fig)
    else:
        return fig

def lines(xs, yss, title='', width=400, height=200, dots=False, legends=None, fig=None, colors=COLORS_AB, alpha=1.0, line_width=1, show=True, **kwargs):
    if fig is None:
        fig = figure(plot_width=width, plot_height=height, tools="save", title=title, **kwargs)

    for i, ys in enumerate(yss):
        legend = None
        if legends is not None:
            legend = legends[i]
        color = None
        if colors is not None and len(colors) >= len(yss):
            color = colors[i]
        fig.line(xs, ys, line_color=color, line_alpha=alpha, line_width=line_width, legend=legend, )
        if dots:
            fig.scatter(xs, ys, line_color=None, fill_color=colors[i], size=4)

    if show:
        plotting.show(fig)
    else:
        return fig
