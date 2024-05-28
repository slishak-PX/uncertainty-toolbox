import numpy as np
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from uncertainty_toolbox.metrics_calibration import (
    get_proportion_lists,
    get_proportion_lists_vectorized,
    adversarial_group_calibration,
)
from uncertainty_toolbox.viz import filter_subset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def plot_intervals(
    y_pred,
    y_std,
    y_true,
    n_subset=None,
    ylims=None,
    num_stds_confidence_bound=2,
    show=True,
):
    """
    Plot predicted values (y_pred) and intervals (y_std) vs observed
    values (y_true).
    """

    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)
    intervals = num_stds_confidence_bound * y_std

    # Plot
    # fig = make_subplots()
    fig = go.Figure(
        data=go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            line=dict(color="#ff7f0e", width=2),
            name="95% Interval",
            error_y=dict(
                color="rgba(255, 153, 0, 0.5)", type="data", array=intervals, visible=True
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="Predicted",
            line=dict(color="#ff7f0e", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ylims, y=ylims, name="Observations", mode="lines", line=dict(color="#1f77b4", width=2)
        )
    )

    fig.update_xaxes(title_text="Observed Values")
    fig.update_yaxes(title_text="<b>Predicted Values and Intervals</b>")
    fig.update_layout(template="none", autosize=True, height=400, width=500)
    fig.layout.font.family = "Arial"
    fig.update_layout(title="Prediction Intervals")

    if ylims:
        fig.update_layout(yaxis_range=[ylims[0], ylims[1]])

    if show:
        fig.show()


def plot_intervals_ordered(
    y_pred,
    y_std,
    y_true,
    n_subset=None,
    ylims=None,
    num_stds_confidence_bound=2,
    show=True,
):
    """
    Plot predicted values (y_pred) and intervals (y_std) vs observed
    values (y_true).
    """
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    order = np.argsort(y_true.flatten())
    y_pred, y_std, y_true = y_pred[order], y_std[order], y_true[order]
    xs = np.arange(len(order))
    intervals = num_stds_confidence_bound * y_std

    # Plot
    fig = go.Figure(
        data=go.Scatter(
            x=xs,
            y=y_pred,
            mode="markers",
            line=dict(color="#ff7f0e", width=2),
            name="95% Interval",
            error_y=dict(
                color="rgba(255, 153, 0, 0.5)", type="data", array=intervals, visible=True
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs, y=y_pred, mode="markers", name="Predicted", line=dict(color="#ff7f0e", width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs, y=y_true, name="Observations", mode="lines", line=dict(color="#1f77b4", width=2)
        )
    )

    # Determine lims
    if ylims is None:
        intervals_lower_upper = [y_pred - intervals, y_pred + intervals]
        lims_ext = [
            int(np.floor(np.min(intervals_lower_upper[0]))),
            int(np.ceil(np.max(intervals_lower_upper[1]))),
        ]
    else:
        lims_ext = ylims

    fig.update_xaxes(title_text="Index (Ordered by Observed Value)")
    fig.update_yaxes(title_text="<b>Predicted Values and Intervals</b>")
    fig.update_layout(template="none", autosize=True, height=400, width=500)
    fig.layout.font.family = "Arial"
    fig.update_layout(title="Ordered Prediction Intervals")

    if ylims:
        fig.update_layout(yaxis_range=[lims_ext[0], lims_ext[1]])

    if show:
        fig.show()


def plot_xy(
    y_pred,
    y_std,
    y_true,
    x,
    n_subset=None,
    ylims=None,
    xlims=None,
    num_stds_confidence_bound=2,
    show=True,
):
    """Plot 1D input (x) and predicted/true (y_pred/y_true) values."""
    if n_subset is not None:
        [y_pred, y_std, y_true, x] = filter_subset([y_pred, y_std, y_true, x], n_subset)

    intervals = num_stds_confidence_bound * y_std

    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=x, y=y_pred, mode="markers", name="Predicted", line=dict(color="#ff7f0e", width=2)
        )
    )
    fig.add_trace(
        go.Scatter(x=x, y=y_true, mode="markers", name="True", line=dict(color="#1f77b4", width=2))
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred - intervals,
            mode="lines",
            fill="none",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred + intervals,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255, 153, 0, 0.2)",
            line=dict(color="rgba(0,0,0,0.0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Interval",
        )
    )

    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="<b>y</b>")
    fig.update_layout(template="none", autosize=True, height=400, width=500)
    fig.layout.font.family = "Arial"
    fig.update_layout(title="Confidence Band")

    if ylims:
        fig.update_layout(yaxis_range=[ylims[0], ylims[1]])

    if show:
        fig.show()


def plot_calibration(
    y_pred,
    y_std,
    y_true,
    n_subset=None,
    curve_label=None,
    show=True,
    vectorized=True,
    exp_props=None,
    obs_props=None,
):
    """
    Make calibration plot using predicted mean values (y_pred), predicted std
    values (y_std), and observed values (y_true).
    """
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    if (exp_props is None) or (obs_props is None):
        # Compute exp_proportions and obs_proportions
        if vectorized:
            (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
                y_pred, y_std, y_true
            )
        else:
            (exp_proportions, obs_proportions) = get_proportion_lists(y_pred, y_std, y_true)
    else:
        # If expected and observed proportions are give
        exp_proportions = np.array(exp_props).flatten()
        obs_proportions = np.array(obs_props).flatten()
        if exp_proportions.shape != obs_proportions.shape:
            raise RuntimeError("exp_props and obs_props shape mismatch")

    # Set figure defaults
    fontsize = 12

    # Set label
    if curve_label is None:
        curve_label = "Predictor"
    # Plot
    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Predicted", line=dict(color="#ff7f0e", width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=exp_proportions,
            y=obs_proportions,
            mode="lines",
            name="True",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            fill="none",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=exp_proportions,
            y=obs_proportions,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255, 153, 0, 0.2)",
            line=dict(color="rgba(0,0,0,0.0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Interval",
        )
    )

    fig.update_xaxes(title_text="Predicted proportion in interval")
    fig.update_yaxes(title_text="<b>Observed proportion in interval</b>")
    fig.update_layout(template="none", autosize=True, height=400, width=500)
    fig.layout.font.family = "Arial"
    fig.update_layout(title="Average Calibration")

    buff = 0.01
    fig.update_layout(yaxis_range=[0 - buff, 1 + buff])

    # Compute miscalibration area
    polygon_points = []
    for point in zip(exp_proportions, obs_proportions):
        polygon_points.append(point)
    for point in zip(reversed(exp_proportions), reversed(exp_proportions)):
        polygon_points.append(point)
    polygon_points.append((exp_proportions[0], obs_proportions[0]))
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy  # original data
    ls = LineString(np.c_[x, y])  # closed, non-simple
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list = [poly.area for poly in polygonize(mls)]
    miscalibration_area = np.asarray(polygon_area_list).sum()

    fig.add_annotation(
        x=1,
        y=0.1,
        text="Miscalibration area = %.2f" % miscalibration_area,
        showarrow=False,
        arrowhead=1,
    )
    if show:
        fig.show()


def plot_calibration_alt(
    y_pred,
    y_std,
    y_true,
    n_subset=None,
    curve_label=None,
    show=True,
    vectorized=True,
    exp_props=None,
    obs_props=None,
):
    """
    Make calibration plot using predicted mean values (y_pred), predicted std
    values (y_std), and observed values (y_true).
    """
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    if (exp_props is None) or (obs_props is None):
        # Compute exp_proportions and obs_proportions
        if vectorized:
            (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
                y_pred, y_std, y_true
            )
        else:
            (exp_proportions, obs_proportions) = get_proportion_lists(y_pred, y_std, y_true)
    else:
        # If expected and observed proportions are give
        exp_proportions = np.array(exp_props).flatten()
        obs_proportions = np.array(obs_props).flatten()
        if exp_proportions.shape != obs_proportions.shape:
            raise RuntimeError("exp_props and obs_props shape mismatch")

    # Set figure defaults
    fontsize = 12

    # Set label
    if curve_label is None:
        curve_label = "Predictor"
    # Plot
    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Predicted", line=dict(color="#ff7f0e", width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=exp_proportions,
            y=obs_proportions,
            mode="lines",
            name="True",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            fill="none",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=exp_proportions,
            y=obs_proportions,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255, 153, 0, 0.2)",
            line=dict(color="rgba(0,0,0,0.0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Interval",
        )
    )

    fig.update_layout(template="none", autosize=True, height=500, width=530)
    fig.layout.font.family = "Georgia"
    fig.update_layout(title="Average Calibration")

    buff = 0.01
    fig.update_layout(xaxis_range=[0 - buff, 1 + buff])
    fig.update_layout(yaxis_range=[0 - buff, 1 + buff])

    fig.update_xaxes(
        title_text="Predicted proportion in interval", tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    fig.update_yaxes(
        title_text="Observed proportion in interval",
        scaleanchor="x",
        scaleratio=1,
        tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )

    # Compute miscalibration area
    polygon_points = []
    for point in zip(exp_proportions, obs_proportions):
        polygon_points.append(point)
    for point in zip(reversed(exp_proportions), reversed(exp_proportions)):
        polygon_points.append(point)
    polygon_points.append((exp_proportions[0], obs_proportions[0]))
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy  # original data
    ls = LineString(np.c_[x, y])  # closed, non-simple
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list = [poly.area for poly in polygonize(mls)]
    miscalibration_area = np.asarray(polygon_area_list).sum()

    fig.add_annotation(
        x=0.77,
        y=0.1,
        text="Miscalibration area = %.2f" % miscalibration_area,
        showarrow=False,
        arrowhead=1,
    )
    if show:
        fig.show()
