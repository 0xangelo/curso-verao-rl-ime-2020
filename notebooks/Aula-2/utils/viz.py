from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import row


output_notebook()


def plot_returns(timesteps, total_rewards, avg_total_rewards):
    p1 = figure(
        title="Avg Returns",
        x_axis_label="Episodes",
        plot_width=600,
        plot_height=400
    )
    p1.line(timesteps, avg_total_rewards)

    p2 = figure(
        title="Returns",
        x_axis_label="Episodes",
        plot_width=600,
        plot_height=400
    )
    p2.line(timesteps, total_rewards)

    show(row([p1, p2]))
