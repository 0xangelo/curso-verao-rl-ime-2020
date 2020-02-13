from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import gridplot, row, column
from bokeh.models import Div
from bokeh.palettes import Spectral6

import numpy as np


output_notebook(hide_banner=True)


def plot_experiments(results, params_dict):
    p1 = figure(
        title="Avg Returns (over last 100 episodes)",
        x_axis_label="Timesteps",
        plot_width=600,
        plot_height=250
    )

    p2 = figure(
        title="Policy Gradient Losses",
        x_axis_label="Timesteps",
        plot_width=600,
        plot_height=250
    )

    xs1, ys1 = [], []
    for timesteps, _, _, avg_total_rewards in results:
        xs1.append(timesteps)
        ys1.append(avg_total_rewards)

    xs2, ys2 = [], []
    for _, losses, _, _ in results:
        timesteps = [t for (t, _) in losses]
        losses = [loss for (_, loss) in losses]
        xs2.append(timesteps)
        ys2.append(losses)

    p1.multi_line(xs1, ys1, color=Spectral6[:len(xs1)])
    p2.multi_line(xs2, ys2, color=Spectral6[:len(xs2)])

    plots = row([p1, p2])

    suptitle = ', '.join([f"{param}={value}" for param, value in params_dict.items()])
    show(column(Div(text=f"<h2>Experiment: {suptitle}</h1>"), plots))


def plot_returns(timesteps, total_rewards, avg_total_rewards):
    p1 = figure(
        title="Returns",
        x_axis_label="Timesteps",
        plot_width=600,
        plot_height=400
    )
    p1.line(timesteps, total_rewards)

    p2 = figure(
        title="Avg Returns (over last 100 episodes)",
        x_axis_label="Timesteps",
        plot_width=600,
        plot_height=400
    )
    p2.line(timesteps, avg_total_rewards)

    show(row([p1, p2]))


def plot_losses(losses):
    timesteps = [t for (t, _) in losses]
    losses = [loss for (_, loss) in losses]
    assert len(timesteps) == len(losses)

    p = figure(
        title="Policy Gradient Loss",
        x_axis_label="Timesteps",
        plot_width=600,
        plot_height=400
    )
    p.line(timesteps, losses)

    show(p)


def plot_gradient_norms(grads):
    figures = {}
    for w_name, _ in grads[0][1]:
        p = figure(
            title=f"grad_{w_name}",
            x_axis_label="Timesteps",
        )
        figures[w_name] = p

    plots = {}
    timesteps = []

    for timestep, gradients in grads:
        timesteps.append(timestep)

        for w_name, grad in gradients:
            plots[w_name] = plots.get(w_name, [])
            plots[w_name].append(grad)

    grid = []
    row = 0
    for i, (w_name, grads) in enumerate(plots.items()):
        p = figures[w_name]
        p.line(timesteps, grads)

        if i % 2 == 0:
            grid.append([])
            grid[row].append(p)
        else:
            grid[row].append(p)
            row += 1

    grid = gridplot(grid, plot_width=500, plot_height=250)
    show(grid)


def plot_action_distribution(agent, batch, n_samples=3000, layout=(3, 4), width=250, height=200):
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
