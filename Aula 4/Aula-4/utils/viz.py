from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import gridplot, row

import numpy as np


output_notebook(hide_banner=True)


def plot_experiments(env, timesteps, total_rewards, avg_total_rewards):
    max_episode_steps = env.spec.max_episode_steps

    total_rewards_mean, total_rewards_std = total_rewards
    avg_total_rewards_mean, avg_total_rewards_std = avg_total_rewards

    p1 = figure(
        title="Returns",
        x_axis_label="Timesteps",
        plot_width=600,
        plot_height=400
    )
    p1.line(timesteps, total_rewards_mean)

    lower = total_rewards_mean - total_rewards_std
    upper = np.clip(total_rewards_mean + total_rewards_std, a_min=None, a_max=max_episode_steps)
    p1.varea(timesteps, y1=lower, y2=upper, fill_alpha=0.25)

    p2 = figure(
        title="Avg Returns (últimos 100 episódios)",
        x_axis_label="Timesteps",
        plot_width=600,
        plot_height=400
    )
    p2.line(timesteps, avg_total_rewards_mean)

    lower = avg_total_rewards_mean - avg_total_rewards_std
    upper = np.clip(avg_total_rewards_mean + avg_total_rewards_std, a_min=None, a_max=max_episode_steps)
    p2.varea(timesteps, y1=lower, y2=upper, fill_alpha=0.25)

    show(row([p1, p2]))

    
    
def plot_metrics(avg_total_rewards, losses):
    
    plot_width = 350
    plot_height = 300
    
    p1 = figure(
        title="Avg Returns (últimos 100 episódios)",
        x_axis_label="Timesteps",
        plot_width=1050,
        plot_height=400
    )
    x = [t for (_, t, _) in avg_total_rewards]
    y = [avg_return for (_, _, avg_return) in avg_total_rewards]
    p1.line(x, y)
    
    show(p1)
    
    p2 = figure(
        title="Policy Loss",
        x_axis_label="Timesteps",
        plot_width=plot_width,
        plot_height=plot_height
    )
    x = [t for (t, _) in losses]
    y = [loss["policy_loss"] for (_, loss) in losses]
    p2.line(x, y)
    
    p3 = figure(
        title="Value Fn Loss",
        x_axis_label="Timesteps",
        plot_width=plot_width,
        plot_height=plot_height
    )
    x = [t for (t, _) in losses]
    y = [loss["vf_loss"] for (_, loss) in losses]
    p3.line(x, y)

    p4 = figure(
        title="Entropy loss",
        x_axis_label="Timesteps",
        plot_width=plot_width,
        plot_height=plot_height
    )
    x = [t for (t, _) in losses]
    y = [loss["entropy_loss"] for (_, loss) in losses]
    p4.line(x, y)
    
    show(row([p2, p3, p4]))
    

def plot_returns(timesteps, total_rewards, avg_total_rewards):
    p1 = figure(
        title="Returns",
        x_axis_label="Timesteps",
        plot_width=600,
        plot_height=400
    )
    p1.line(timesteps, total_rewards)

    p2 = figure(
        title="Avg Returns (últimos 100 episódios)",
        x_axis_label="Timesteps",
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