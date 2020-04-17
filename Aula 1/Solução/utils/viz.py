from bokeh.io import output_notebook
from bokeh.layouts import gridplot, row
from bokeh.models import Span
from bokeh.plotting import figure, show

import numpy as np


output_notebook()


def plot_action_distribution(agent, n_samples=1000, layout=(2, 4), width=250, height=250):
    grid = []
    for i in range(layout[0]):
        grid.append([])
        for j in range(layout[1]):
            obs = agent.obs_space.sample()
            actions = [agent.act(obs) for _ in range(n_samples)]

            hist, edges = np.histogram(actions, density=True, bins=agent.action_space.n)
            
            p = figure(
                title=f"obs = {obs}",
                x_axis_label="Actions",
                y_axis_label="Action Distribution",
            )
            p.xgrid.visible = False

            tickers = (edges[:-1] + edges[1:]) / 2
            p.xaxis.ticker = tickers
            
            p.xaxis.major_label_overrides = dict(zip(
                tickers,
                map(str, range(agent.action_space.n))))
            
            p.quad(
                top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white", alpha=0.5)

            grid[i].append(p)
    
    grid = gridplot(grid, plot_width=width, plot_height=height)
    show(grid)


def plot_episode_total_rewards(returns, width=500, height=350):
    avg_return = np.mean(returns)
    
    p1 = figure(
        title="Episode Return",
        y_axis_label="Total Reward",
        x_axis_label="Episodes",
        width=width,
        height=height,
        toolbar_location="above"
    )
    p1.line(x=range(len(returns)), y=returns)
    hline = Span(location=avg_return, dimension="width", line_color="red", line_width=1)
    p1.renderers.append(hline)
    
    p2 = figure(
        title="Episode Return (Histogram)",
        y_axis_label="Total Reward",
        width=width,
        height=height,
        toolbar_location="above"
    )
    hist, edges = np.histogram(returns, density=True, bins=50)
    p2.quad(
        top=hist, bottom=0, left=edges[:-1], right=edges[1:]
        , line_color="white", alpha=0.5)
    vline = Span(location=avg_return, dimension="height", line_color="red", line_width=1)
    p2.renderers.append(vline)

    show(row(p1, p2))

    
