from __future__ import annotations

import json
import sys
from datetime import datetime, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


V4_ROOT = Path(__file__).resolve().parent
V4_SRC = V4_ROOT / "src"
if str(V4_SRC) not in sys.path:
    sys.path.insert(0, str(V4_SRC))


MODEL_PATH = V4_ROOT / "artifacts" / "csi1000_v4_final_model.joblib"
REPLAY_PATH = (
    V4_ROOT / "artifacts" / "historical_replay_predictions.joblib"
)
REPORT_DIR = V4_ROOT / "reports"
CONFIG_PATH = V4_ROOT / "config" / "final_config.json"

ROUTE_LABELS = {
    "preopen": "盘前",
    "early": "开盘首小时",
    "late": "首小时后",
}
ROUTE_COLORS = {
    "preopen": "#7C3AED",
    "early": "#0E7490",
    "late": "#1D4ED8",
}
QUANTILE_COLUMNS = [
    f"q{value:02d}_final_amp" for value in range(10, 100, 10)
]
PROBABILITY_COLUMNS = [
    "prob_final_amp_gt_1_0pct",
    "prob_final_amp_gt_1_5pct",
    "prob_final_amp_gt_2_0pct",
    "prob_final_amp_gt_3_0pct",
    "prob_final_amp_gt_4_0pct",
]
PROBABILITY_LABELS = {
    "prob_final_amp_gt_1_0pct": "> 1.0%",
    "prob_final_amp_gt_1_5pct": "> 1.5%",
    "prob_final_amp_gt_2_0pct": "> 2.0%",
    "prob_final_amp_gt_3_0pct": "> 3.0%",
    "prob_final_amp_gt_4_0pct": "> 4.0%",
}


st.set_page_config(
    page_title="中证1000振幅动态概率模型",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #10233f;
        --muted: #60738f;
        --line: #dfe7f1;
        --panel: #ffffff;
        --teal: #0f8b8d;
        --blue: #1d4ed8;
    }
    .stApp {
        background:
            radial-gradient(circle at 92% 2%, rgba(15,139,141,.09), transparent 24rem),
            #f5f8fc;
    }
    .block-container {
        max-width: 1480px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: linear-gradient(115deg, #0b1f3a 0%, #123f5c 58%, #0f8b8d 100%);
        border-radius: 22px;
        padding: 1.55rem 1.8rem 1.45rem;
        color: white;
        box-shadow: 0 18px 42px rgba(18, 52, 86, .18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0 0 .35rem 0;
        font-size: 2rem;
        letter-spacing: .02em;
    }
    .hero p {
        margin: 0;
        color: rgba(255,255,255,.78);
        font-size: .98rem;
    }
    .eyebrow {
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        color: #7dd3fc;
        margin-bottom: .45rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.94);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: .85rem 1rem;
        box-shadow: 0 7px 22px rgba(26, 60, 95, .06);
    }
    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }
    div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 720;
    }
    .status-row {
        display: flex;
        gap: .55rem;
        align-items: center;
        flex-wrap: wrap;
        margin: .1rem 0 1rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: .35rem;
        padding: .32rem .7rem;
        border-radius: 999px;
        background: #e7f7f6;
        color: #116466;
        font-size: .8rem;
        font-weight: 650;
        border: 1px solid #bfe7e5;
    }
    .status-pill.secondary {
        background: #eef3fb;
        color: #294f83;
        border-color: #d8e3f3;
    }
    .section-note {
        color: var(--muted);
        font-size: .86rem;
        margin-top: -.5rem;
        margin-bottom: .8rem;
    }
    div[data-testid="stTabs"] button {
        font-weight: 650;
    }
    section[data-testid="stSidebar"] {
        background: #eef3f9;
        border-right: 1px solid #dce5ef;
    }
    [data-testid="stToolbar"] {
        display: none;
    }
    .small-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: .9rem 1rem;
        color: var(--ink);
    }
    .small-card .label {
        color: var(--muted);
        font-size: .78rem;
        margin-bottom: .25rem;
    }
    .small-card .value {
        font-size: 1.22rem;
        font-weight: 730;
    }
    .footer {
        color: #71839b;
        text-align: center;
        font-size: .78rem;
        padding-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_replay_payload() -> dict:
    return joblib.load(REPLAY_PATH)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_datetime(date_value: pd.Timestamp, bar_index: int) -> pd.Timestamp:
    date_value = pd.Timestamp(date_value).normalize()
    bar_index = int(bar_index)
    if bar_index == 0:
        return date_value + pd.Timedelta(hours=9, minutes=25)
    if bar_index <= 120:
        return date_value + pd.Timedelta(hours=9, minutes=30 + bar_index)
    return date_value + pd.Timedelta(
        hours=13,
        minutes=bar_index - 120,
    )


def checkpoint_label(bar_index: int) -> str:
    anchor = pd.Timestamp.combine(pd.Timestamp("2000-01-01"), time(0, 0))
    return checkpoint_datetime(anchor, int(bar_index)).strftime("%H:%M")


def prepare_day_frame(frame: pd.DataFrame, selected_date: pd.Timestamp) -> pd.DataFrame:
    day = frame.loc[frame["date"].eq(selected_date)].copy()
    day["timestamp"] = [
        checkpoint_datetime(selected_date, value)
        for value in day["bar_index"]
    ]
    day["time_label"] = day["timestamp"].dt.strftime("%H:%M")
    day["route_label"] = day["route"].map(ROUTE_LABELS)
    return day.sort_values("bar_index").reset_index(drop=True)


def make_fan_chart(
    day: pd.DataFrame,
    selected_bar: int,
    show_actual: bool,
) -> go.Figure:
    fig = go.Figure()
    bands = [
        ("q10_final_amp", "q90_final_amp", "rgba(29,78,216,.08)", "P10–P90"),
        ("q20_final_amp", "q80_final_amp", "rgba(29,78,216,.10)", "P20–P80"),
        ("q30_final_amp", "q70_final_amp", "rgba(15,139,141,.12)", "P30–P70"),
        ("q40_final_amp", "q60_final_amp", "rgba(15,139,141,.18)", "P40–P60"),
    ]
    for lower, upper, color, label in bands:
        fig.add_trace(
            go.Scatter(
                x=day["timestamp"],
                y=day[lower],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=day["timestamp"],
                y=day[upper],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=color,
                name=label,
                hovertemplate=label + " 上界 %{y:.3f}%<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=day["timestamp"],
            y=day["q50_final_amp"],
            mode="lines",
            name="最终振幅 P50",
            line={"color": "#0f8b8d", "width": 3},
            hovertemplate="%{x|%H:%M}<br>P50 %{y:.3f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=day["timestamp"],
            y=day["current_amp"],
            mode="lines",
            name="当前已实现振幅",
            line={"color": "#f59e0b", "width": 2.5},
            hovertemplate="%{x|%H:%M}<br>已实现 %{y:.3f}%<extra></extra>",
        )
    )
    if show_actual:
        fig.add_trace(
            go.Scatter(
                x=day["timestamp"],
                y=day["final_amp"],
                mode="lines",
                name="实际最终振幅",
                line={"color": "#dc2626", "width": 2, "dash": "dash"},
                hovertemplate="实际最终振幅 %{y:.3f}%<extra></extra>",
            )
        )
    selected_time = day.loc[
        day["bar_index"].eq(selected_bar), "timestamp"
    ].iloc[0]
    fig.add_vline(
        x=selected_time.to_pydatetime(),
        line_width=2,
        line_dash="dot",
        line_color="#7c3aed",
    )
    split_time = checkpoint_datetime(day["date"].iloc[0], 60)
    fig.add_vline(
        x=split_time.to_pydatetime(),
        line_width=1,
        line_dash="dash",
        line_color="#94a3b8",
    )
    lunch_start = pd.Timestamp(day["date"].iloc[0]) + pd.Timedelta(
        hours=11, minutes=30
    )
    lunch_end = pd.Timestamp(day["date"].iloc[0]) + pd.Timedelta(hours=13)
    fig.add_vrect(
        x0=lunch_start.to_pydatetime(),
        x1=lunch_end.to_pydatetime(),
        fillcolor="rgba(148,163,184,.08)",
        line_width=0,
        annotation_text="午休",
        annotation_position="top left",
    )
    fig.update_layout(
        title={"text": "最终振幅动态分位数带", "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=500,
        margin={"l": 20, "r": 15, "t": 55, "b": 20},
        hovermode="x unified",
        yaxis={"title": "振幅（%）", "rangemode": "tozero"},
        xaxis={"title": None, "tickformat": "%H:%M"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    return fig


def make_probability_chart(day: pd.DataFrame, selected_bar: int) -> go.Figure:
    colors = ["#0f8b8d", "#2563eb", "#7c3aed", "#ea580c", "#dc2626"]
    fig = go.Figure()
    for column, color in zip(PROBABILITY_COLUMNS, colors):
        fig.add_trace(
            go.Scatter(
                x=day["timestamp"],
                y=day[column] * 100,
                mode="lines",
                name=PROBABILITY_LABELS[column],
                line={"width": 2.2, "color": color},
                hovertemplate=(
                    "%{x|%H:%M}<br>"
                    + PROBABILITY_LABELS[column]
                    + "：%{y:.1f}%<extra></extra>"
                ),
            )
        )
    selected_time = day.loc[
        day["bar_index"].eq(selected_bar), "timestamp"
    ].iloc[0]
    fig.add_vline(
        x=selected_time.to_pydatetime(),
        line_width=2,
        line_dash="dot",
        line_color="#7c3aed",
    )
    fig.update_layout(
        title={"text": "指定振幅超越概率", "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=410,
        margin={"l": 20, "r": 15, "t": 55, "b": 20},
        hovermode="x unified",
        yaxis={"title": "概率（%）", "range": [0, 100]},
        xaxis={"title": None, "tickformat": "%H:%M"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    return fig


def make_quantile_snapshot(selected: pd.Series, show_actual: bool) -> go.Figure:
    levels = list(range(10, 100, 10))
    values = [selected[f"q{value:02d}_final_amp"] for value in levels]
    fig = go.Figure(
        go.Scatter(
            x=levels,
            y=values,
            mode="lines+markers",
            line={"color": "#1d4ed8", "width": 3},
            marker={"size": 9, "color": "#0f8b8d"},
            name="条件分位数",
            hovertemplate="P%{x}<br>%{y:.3f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=float(selected["current_amp"]),
        line_color="#f59e0b",
        line_dash="dot",
        annotation_text="当前已实现",
    )
    if show_actual:
        fig.add_hline(
            y=float(selected["final_amp"]),
            line_color="#dc2626",
            line_dash="dash",
            annotation_text="实际最终",
        )
    fig.update_layout(
        title={"text": "当前时点分位数分布", "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=410,
        margin={"l": 20, "r": 15, "t": 55, "b": 20},
        xaxis={"title": "分位数", "tickprefix": "P", "dtick": 10},
        yaxis={"title": "最终振幅（%）", "rangemode": "tozero"},
        showlegend=False,
    )
    return fig


def make_history_chart(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["final_amp"],
            mode="lines",
            name="每日最终振幅",
            line={"color": "rgba(37,99,235,.30)", "width": 1},
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["rolling_median_20"],
            mode="lines",
            name="20日中位数",
            line={"color": "#0f8b8d", "width": 2.5},
            hovertemplate="%{x|%Y-%m-%d}<br>20日中位数 %{y:.3f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "历史振幅与20日中位数", "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=450,
        margin={"l": 20, "r": 15, "t": 55, "b": 20},
        yaxis={"title": "振幅（%）", "rangemode": "tozero"},
        xaxis={"title": None},
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    return fig


def make_route_metric_chart(route_metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=route_metrics["route_label"],
            y=route_metrics["q3_raw_pinball"],
            name="P10/P50/P90 Loss",
            marker_color="#0f8b8d",
            text=route_metrics["q3_raw_pinball"].map(lambda value: f"{value:.4f}"),
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=route_metrics["route_label"],
            y=route_metrics["all9_raw_pinball"],
            name="P10–P90 Loss",
            marker_color="#1d4ed8",
            text=route_metrics["all9_raw_pinball"].map(lambda value: f"{value:.4f}"),
            textposition="outside",
        )
    )
    fig.update_layout(
        title={"text": "分路由Pinball Loss", "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        height=410,
        barmode="group",
        margin={"l": 20, "r": 15, "t": 55, "b": 20},
        yaxis={"title": "Loss", "rangemode": "tozero"},
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    return fig


def model_tree_table(bundle: dict) -> pd.DataFrame:
    rows = []
    for route, expert in bundle["routes"].items():
        row = {"路由": ROUTE_LABELS[route]}
        row.update(
            {
                suffix.upper(): int(iteration)
                for suffix, iteration in expert["iterations"].items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def probability_card(label: str, value: float) -> str:
    return (
        '<div class="small-card">'
        f'<div class="label">P(最终振幅 {label})</div>'
        f'<div class="value">{value:.1%}</div>'
        "</div>"
    )


if not MODEL_PATH.exists() or not REPLAY_PATH.exists():
    st.error(
        "缺少模型或历史回放数据。请先运行 scripts/build_replay_data.py。"
    )
    st.stop()

bundle = load_model_bundle()
payload = load_replay_payload()
replay = payload["frame"].copy()
replay["date"] = pd.to_datetime(replay["date"]).dt.normalize()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">CSI 1000 · Intraday Amplitude Intelligence</div>
      <h1>中证1000当天振幅动态概率模型</h1>
      <p>盘前与盘中每5分钟更新最终振幅分位数和指定振幅超越概率</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 回放控制")
    years = sorted(replay["date"].dt.year.unique(), reverse=True)
    selected_year = st.selectbox("年份", years, index=0)
    year_dates = sorted(
        replay.loc[replay["date"].dt.year.eq(selected_year), "date"].unique(),
        reverse=True,
    )
    selected_date = pd.Timestamp(
        st.selectbox(
            "交易日",
            year_dates,
            format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"),
        )
    )
    day = prepare_day_frame(replay, selected_date)
    bar_options = day["bar_index"].astype(int).tolist()
    default_bar = 60 if 60 in bar_options else bar_options[-1]
    selected_bar = st.select_slider(
        "预测时点",
        options=bar_options,
        value=default_bar,
        format_func=checkpoint_label,
    )
    show_actual = st.toggle("显示实际最终振幅", value=True)
    st.divider()
    st.markdown("### 模型口径")
    st.caption("振幅 = (日内最高价 − 日内最低价) / 前收盘价 × 100")
    st.caption("目标指数：中证1000")
    st.caption("因子指数：上证50、沪深300")
    st.caption(f"回放范围：{payload['date_start']} 至 {payload['date_end']}")

selected = day.loc[day["bar_index"].eq(selected_bar)].iloc[0]
route_label = ROUTE_LABELS[selected["route"]]
time_label = selected["time_label"]
period_label = selected["period_label"]

st.markdown(
    (
        '<div class="status-row">'
        f'<span class="status-pill">● {period_label}</span>'
        f'<span class="status-pill secondary">{selected_date:%Y-%m-%d} · {time_label}</span>'
        f'<span class="status-pill secondary" style="border-color:{ROUTE_COLORS[selected["route"]]};">'
        f"{route_label}</span>"
        "</div>"
    ),
    unsafe_allow_html=True,
)

tab_replay, tab_history, tab_model, tab_audit, tab_info = st.tabs(
    [
        "动态回放",
        "历史行情",
        "模型表现",
        "数据审计",
        "模型说明",
    ]
)

with tab_replay:
    columns = st.columns([1, 1.15, 1.45, 1])
    columns[0].metric("当前已实现振幅", f"{selected['current_amp']:.3f}%")
    columns[1].metric(
        "最终振幅 P50",
        f"{selected['q50_final_amp']:.3f}%",
        f"预计剩余 {selected['q50_final_amp'] - selected['current_amp']:.3f}%",
    )
    columns[2].metric(
        "P10–P90区间",
        f"{selected['q10_final_amp']:.2f}%–{selected['q90_final_amp']:.2f}%",
    )
    columns[3].metric(
        "实际最终振幅",
        f"{selected['final_amp']:.3f}%" if show_actual else "已隐藏",
    )

    st.markdown("#### 指定振幅超越概率")
    probability_columns = st.columns(5)
    for output, column in zip(probability_columns, PROBABILITY_COLUMNS):
        output.markdown(
            probability_card(
                PROBABILITY_LABELS[column],
                float(selected[column]),
            ),
            unsafe_allow_html=True,
        )

    st.plotly_chart(
        make_fan_chart(day, selected_bar, show_actual),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": False},
    )
    left, right = st.columns([1.25, 1])
    with left:
        st.plotly_chart(
            make_probability_chart(day, selected_bar),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with right:
        st.plotly_chart(
            make_quantile_snapshot(selected, show_actual),
            use_container_width=True,
            config={"displaylogo": False},
        )

    with st.expander("查看并下载当日48个预测点"):
        display_columns = [
            "date",
            "time_label",
            "route_label",
            "current_amp",
            *QUANTILE_COLUMNS,
            *PROBABILITY_COLUMNS,
        ]
        display = day[display_columns].copy().rename(
            columns={
                "date": "日期",
                "time_label": "时点",
                "route_label": "路由",
                "current_amp": "当前振幅",
                **{
                    column: column.split("_")[0].upper()
                    for column in QUANTILE_COLUMNS
                },
                **{
                    column: f"P(振幅{PROBABILITY_LABELS[column]})"
                    for column in PROBABILITY_COLUMNS
                },
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button(
            "下载当日预测CSV",
            data=display.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"CSI1000_amplitude_{selected_date:%Y%m%d}.csv",
            mime="text/csv",
        )

with tab_history:
    daily = (
        replay.loc[replay["bar_index"].eq(0), ["date", "final_amp"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["year"] = daily["date"].dt.year
    daily["rolling_median_20"] = daily["final_amp"].rolling(
        20, min_periods=5
    ).median()
    yearly = (
        daily.groupby("year", as_index=False)
        .agg(
            交易日=("date", "size"),
            平均振幅=("final_amp", "mean"),
            中位振幅=("final_amp", "median"),
            P90振幅=("final_amp", lambda values: values.quantile(0.90)),
            最大振幅=("final_amp", "max"),
            超过2比例=("final_amp", lambda values: values.gt(2.0).mean()),
            超过3比例=("final_amp", lambda values: values.gt(3.0).mean()),
        )
        .sort_values("year", ascending=False)
    )
    latest_daily = daily.iloc[-1]
    history_metrics = st.columns(4)
    history_metrics[0].metric("历史交易日", f"{len(daily):,}")
    history_metrics[1].metric("全样本中位振幅", f"{daily['final_amp'].median():.3f}%")
    history_metrics[2].metric("全样本P90振幅", f"{daily['final_amp'].quantile(.90):.3f}%")
    history_metrics[3].metric("最新交易日振幅", f"{latest_daily['final_amp']:.3f}%")
    st.plotly_chart(
        make_history_chart(daily),
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.markdown("#### 年度振幅分布")
    display_yearly = yearly.rename(columns={"year": "年份"}).copy()
    for column in ("平均振幅", "中位振幅", "P90振幅", "最大振幅"):
        display_yearly[column] = display_yearly[column].map(
            lambda value: f"{value:.3f}%"
        )
    for column in ("超过2比例", "超过3比例"):
        display_yearly[column] = display_yearly[column].map(
            lambda value: f"{value:.1%}"
        )
    st.dataframe(display_yearly, use_container_width=True, hide_index=True)
    st.caption("历史行情页用于观察振幅分布；模型正式评价指标见“模型表现”页。")

with tab_model:
    model_summary = load_csv(str(REPORT_DIR / "model_summary.csv"))
    route_metrics = load_csv(str(REPORT_DIR / "route_metrics.csv"))
    route_metrics["route_label"] = route_metrics["route"].map(ROUTE_LABELS)
    summary_map = dict(zip(model_summary["metric"], model_summary["value"]))
    model_metrics = st.columns(4)
    model_metrics[0].metric(
        "主Loss",
        f"{summary_map['primary_route_equal_q3_raw_pinball']:.6f}",
    )
    model_metrics[1].metric(
        "单调排序后主Loss",
        f"{summary_map['route_equal_q3_sorted_pinball']:.6f}",
    )
    model_metrics[2].metric(
        "九分位数路由等权Loss",
        f"{summary_map['route_equal_all9_raw_pinball']:.6f}",
    )
    model_metrics[3].metric("评价交易日", f"{int(route_metrics['days'].max())}")
    st.markdown(
        '<div class="section-note">主指标：三个路由等权，P10、P50、P90等权。</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(
            make_route_metric_chart(route_metrics),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with right:
        quality = go.Figure()
        quality.add_trace(
            go.Bar(
                x=route_metrics["route_label"],
                y=route_metrics["p10_p90_coverage"] * 100,
                name="P10–P90覆盖率",
                marker_color="#0f8b8d",
                text=(route_metrics["p10_p90_coverage"] * 100).map(
                    lambda value: f"{value:.1f}%"
                ),
                textposition="outside",
            )
        )
        quality.add_hline(
            y=80,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text="目标80%",
        )
        quality.update_layout(
            title={"text": "分路由区间覆盖率", "x": 0.01, "xanchor": "left"},
            template="plotly_white",
            height=410,
            margin={"l": 20, "r": 15, "t": 55, "b": 20},
            yaxis={"title": "覆盖率（%）", "range": [0, 100]},
            showlegend=False,
        )
        st.plotly_chart(
            quality,
            use_container_width=True,
            config={"displaylogo": False},
        )
    route_display = route_metrics[
        [
            "route_label",
            "q3_raw_pinball",
            "all9_raw_pinball",
            "p50_mae_amp_pct_points",
            "p10_p90_coverage",
        ]
    ].rename(
        columns={
            "route_label": "路由",
            "q3_raw_pinball": "P10/P50/P90 Loss",
            "all9_raw_pinball": "P10–P90 Loss",
            "p50_mae_amp_pct_points": "P50 MAE（百分点）",
            "p10_p90_coverage": "P10–P90覆盖率",
        }
    )
    st.dataframe(route_display, use_container_width=True, hide_index=True)
    st.markdown("#### 最终树数")
    st.dataframe(model_tree_table(bundle), use_container_width=True, hide_index=True)

with tab_audit:
    audit = load_csv(str(REPORT_DIR / "data_audit.csv"))
    anomalies = load_json(str(REPORT_DIR / "data_anomalies.json"))
    audit_metrics = st.columns(4)
    audit_metrics[0].metric("使用数据源", f"{len(audit)}组")
    audit_metrics[1].metric("日线记录", f"{int(audit['daily_rows'].sum()):,}")
    audit_metrics[2].metric("分钟记录", f"{int(audit['minute_rows'].sum()):,}")
    audit_metrics[3].metric("缺失值", f"{int(audit['missing_values'].sum())}")
    st.markdown("#### 数据完整性")
    audit_display = audit.rename(
        columns={
            "dataset": "数据集",
            "label": "名称",
            "daily_rows": "日线数",
            "minute_rows": "分钟线数",
            "start": "开始日期",
            "end": "结束日期",
            "daily_duplicates": "日线重复",
            "minute_duplicates": "分钟重复",
            "missing_values": "缺失值",
            "min_bars_per_day": "每日最少分钟数",
            "max_bars_per_day": "每日最多分钟数",
            "invalid_ohlc_repaired": "OHLC修复数",
            "timestamps_aligned": "时间戳对齐",
        }
    )
    st.dataframe(audit_display, use_container_width=True, hide_index=True)
    st.markdown("#### 异常处理记录")
    if anomalies:
        for anomaly in anomalies:
            dataset = anomaly.get("dataset", "unknown")
            anomaly_type = anomaly.get("type", "unknown")
            action = anomaly.get("action", "")
            st.markdown(
                f"- **{dataset} · {anomaly_type}**：{action}"
            )
    else:
        st.success("未记录需要处理的数据异常。")

with tab_info:
    st.markdown("### 预测逻辑")
    st.markdown("模型预测目标为：")
    st.latex(
        r"y=\log\left(1+\max\left("
        r"A_{\mathrm{final}}-A_{\mathrm{current}},0\right)\right)"
    )
    st.markdown("振幅统一采用：")
    st.latex(
        r"Amplitude=\frac{High-Low}{PreClose}\times 100"
    )
    st.markdown(
        "每个预测时点由当前已实现振幅加上预测的剩余振幅，"
        "得到最终振幅P10至P90。"
    )
    st.markdown("### 三个时间路由")
    route_info = pd.DataFrame(
        [
            {
                "路由": "盘前",
                "范围": "09:25，bar_index=0",
                "模型数": 9,
                "信息": "仅使用上一交易日及更早历史因子",
            },
            {
                "路由": "开盘首小时",
                "范围": "09:35–10:30",
                "模型数": 9,
                "信息": "使用截至当前时点的日内动态因子",
            },
            {
                "路由": "首小时后",
                "范围": "10:35–11:30、13:05–14:55",
                "模型数": 9,
                "信息": "使用截至当前时点的日内动态因子",
            },
        ]
    )
    st.dataframe(route_info, use_container_width=True, hide_index=True)
    st.markdown("### 模型信息")
    model_info = {
        "模型名称": bundle["model_name"],
        "训练区间": f"{bundle['trained_from']} 至 {bundle['trained_through']}",
        "目标指数": "中证1000",
        "外部因子": "上证50、沪深300",
        "子模型数量": "27",
        "条件分位数": "P10、P20、…、P90",
        "早停阈值": f"{bundle['early_stopping']['min_delta']:.0e}",
        "最终树数来源": "三个120交易日窗口最佳树数中位数",
    }
    st.dataframe(
        pd.DataFrame(model_info.items(), columns=["项目", "内容"]),
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        "2019—2025用于历史回放展示；模型表现页使用现有正式评价报告。"
    )

st.markdown(
    '<div class="footer">中证1000当天振幅动态概率模型 V4 · 本地演示版</div>',
    unsafe_allow_html=True,
)
