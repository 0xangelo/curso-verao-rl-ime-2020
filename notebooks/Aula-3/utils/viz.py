from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import gridplot, row

import numpy as np


output_notebook()


def plot_returns(timesteps, total_rewards, avg_total_rewards):
    p1 = figure(
        title="Returns",
        x_axis_label="Episodes",
        plot_width=600,
        plot_height=400
    )
    p1.line(timesteps, total_rewards)

    p2 = figure(
        title="Avg Returns (últimos 100 episódios)",
        x_axis_label="Episodes",
        plot_width=600,
        plot_height=400
    )
    p2.line(timesteps, avg_total_rewards)

    show(row([p1, p2]))


def plot_action_distribution(agent, n_samples=3000, layout=(3, 4), width=250, height=200):
    batch = agent.memory.sample()
    states = batch["states"]
    
    grid = []
    for i in range(layout[0]):
        grid.append([])
        for j in range(layout[1]):
            
            episode = np.random.randint(low=0, high=len(states))
            timestep = np.random.randint(low=0, high=len(states[episode]))
            obs = states[episode][timestep]

            actions = [agent.act(obs) for _ in range(n_samples)]
            hist, edges = np.histogram(actions, density=True, bins=agent.action_space.n)
            
            obs = ', '.join([f"{v:.2f}" for v in obs])
            
            p = figure(
                title=f"obs = [{obs}]",
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