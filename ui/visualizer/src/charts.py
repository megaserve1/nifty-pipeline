"""Plotly figure construction: candles + label markers + stacked feature panels.

Everything renders into a single :class:`plotly.graph_objects.Figure` built from
``make_subplots(shared_xaxes=True)`` so the price chart and every feature panel
pan, zoom and hover as one unit -- that shared x-axis is what makes it possible
to read "target here, prediction there, features doing this" in one glance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from . import transforms

# --------------------------------------------------------- royal purple ----
# Surfaces and ink. Every colour below was checked against SURFACE for >= 3:1.
PLANE = "#120B22"        # page behind the figure
SURFACE = "#1B1035"      # the plotting surface itself
PANEL = "#241546"        # cards / hover boxes
INK = "#F5F1FF"          # primary text
INK_SOFT = "#C4B5E8"     # axis titles, secondary text
INK_MUTED = "#9A8BC4"    # tick labels, spikes
GRID = "rgba(154,139,196,0.16)"
AXIS_LINE = "rgba(154,139,196,0.34)"
ACCENT = "#A855F7"       # royal amethyst, matches the Streamlit primary colour

# Status colours -- reserved, never reused as a series colour. Each is paired
# with a distinct marker symbol and a named legend entry, so meaning never rests
# on hue alone.
BULL_COLOR = "#0ca30c"       # good      5.34:1  (labels only)
BEAR_COLOR = "#d03b3b"       # critical  3.73:1  (labels only)
MISMATCH_COLOR = "#fab219"   # warning   9.77:1
FLAT_COLOR = "#8B7FB0"       # neutral   4.91:1

# Candle body colours. Kept separate from the label status colours because the
# candles are a *reading* surface -- warmer, punchier greens and reds read more
# like a trading terminal -- while the label markers are a *coding* system with
# a stricter WCAG budget. Both sets validated on the purple surface (>= 3:1).
CANDLE_BULL = "#26A65B"      # 5.71:1
CANDLE_BEAR = "#EF4C4C"      # 4.96:1

# Categorical series palette, in fixed order -- assigned by slot, never by rank.
# Validated against SURFACE: lightness band, chroma floor, colour-blind
# separation (worst adjacent dE 8.4) and contrast all pass.
PALETTE = (
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
)
# Past the eighth series, hue repeats -- so line dash carries the difference.
# Colour + dash together keep every series distinguishable (composite encoding).
DASHES = ("solid", "dash", "dot", "dashdot")

# Single-hue ramp for the confusion matrix. Monotonic in lightness, and capped
# so the count printed in each cell keeps >= 4:1 against its own fill.
SEQUENTIAL = (
    (0.00, "#241546"), (0.25, "#3A2568"), (0.50, "#513391"),
    (0.75, "#6B44B8"), (1.00, "#8A5CD6"),
)

SYMBOLS = (
    "circle", "square", "diamond", "star", "hexagon",
    "pentagon", "cross", "x", "triangle-left", "triangle-right",
)

TEMPLATE_NAME = "royal_purple"


def _build_template() -> go.layout.Template:
    """Register the royal-purple Plotly template."""
    return go.layout.Template(
        layout={
            "paper_bgcolor": PLANE,
            "plot_bgcolor": SURFACE,
            "colorway": list(PALETTE),
            "font": {"color": INK_SOFT, "family": "system-ui, -apple-system, sans-serif",
                     "size": 12},
            "title": {"font": {"color": INK, "size": 15}},
            "xaxis": {
                "gridcolor": GRID, "zerolinecolor": AXIS_LINE, "linecolor": AXIS_LINE,
                "tickcolor": AXIS_LINE, "tickfont": {"color": INK_MUTED},
                "title": {"font": {"color": INK_SOFT}},
            },
            "yaxis": {
                "gridcolor": GRID, "zerolinecolor": AXIS_LINE, "linecolor": AXIS_LINE,
                "tickcolor": AXIS_LINE, "tickfont": {"color": INK_MUTED},
                "title": {"font": {"color": INK_SOFT}},
            },
            "legend": {"font": {"color": INK_SOFT}, "bgcolor": "rgba(0,0,0,0)"},
            "hoverlabel": {
                "bgcolor": PANEL, "bordercolor": ACCENT,
                "font": {"color": INK, "family": "system-ui, sans-serif", "size": 12},
            },
            "colorscale": {"sequential": [list(stop) for stop in SEQUENTIAL]},
        }
    )


pio.templates[TEMPLATE_NAME] = _build_template()

_BULL_TOKENS = {"buy", "long", "up", "bull", "bullish", "1", "+1", "call", "ce", "b"}
_BEAR_TOKENS = {"sell", "short", "down", "bear", "bearish", "-1", "put", "pe", "s"}
_FLAT_TOKENS = {"hold", "flat", "neutral", "none", "no_trade", "notrade", "0", "wait", "n"}

# The nifty-pipeline 7-class ladder: ENTRY_/EXIT_ carry direction, and
# SUPER > SUB > SMALL carry conviction/size. Direction is encoded twice -- hue
# *and* triangle orientation -- so green/red never has to be told apart by a
# red-green colourblind reader. Every shade clears 3:1 on SURFACE.
_ENTRY_SHADES = {"super": "#22C55E", "sub": "#16A34A", "small": "#0E7A38"}
_EXIT_SHADES = {"super": "#F43F5E", "sub": "#DC2626", "small": "#C62828"}
_TIER_SCALE = {"super": 1.35, "sub": 1.05, "small": 0.80}
_TIERS = ("super", "sub", "small")


def _size_tier(token: str) -> Optional[str]:
    """'entry_super' -> 'super'. ``None`` when the class carries no size tier."""
    return next((tier for tier in _TIERS if tier in token), None)

MARKER_MODES = ("below bar", "above bar", "at close", "at high", "at low")
PRICE_STYLES = ("candlestick", "ohlc bars", "close line")
CHART_TYPES = ("line", "area", "step", "bar")


@dataclass(frozen=True)
class LabelStyle:
    color: str
    symbol: str
    scale: float = 1.0     # multiplies the base marker size


def style_for_labels(classes: Sequence[str]) -> dict[str, LabelStyle]:
    """Assign a colour, marker symbol and relative size to every label class.

    Three naming conventions are understood, in order:

    * ``ENTRY_*`` / ``EXIT_*``  -- green up / red down, sized by the
      SUPER > SUB > SMALL ladder.
    * ``buy``/``sell``/``hold``, ``1``/``-1``/``0`` -- the conventional
      green-up / red-down / grey-flat treatment.
    * anything else cycles the validated categorical palette.
    """
    styles: dict[str, LabelStyle] = {}
    spare = 0
    # `dict.fromkeys` preserves first-seen order and removes duplicates so an
    # input like ["A","A","B"] produces the same styles as ["A","B"] -- without
    # this, a repeat advanced the palette counter and shifted every later
    # class's colour depending on the input's multiplicity.
    for cls in dict.fromkeys(classes):
        token = str(cls).strip().lower()
        tier = _size_tier(token)

        if token.startswith("entry"):
            styles[cls] = LabelStyle(
                _ENTRY_SHADES.get(tier, BULL_COLOR), "triangle-up",
                _TIER_SCALE.get(tier, 1.0),
            )
        elif token.startswith("exit"):
            styles[cls] = LabelStyle(
                _EXIT_SHADES.get(tier, BEAR_COLOR), "triangle-down",
                _TIER_SCALE.get(tier, 1.0),
            )
        elif token in _BULL_TOKENS:
            styles[cls] = LabelStyle(BULL_COLOR, "triangle-up")
        elif token in _BEAR_TOKENS:
            styles[cls] = LabelStyle(BEAR_COLOR, "triangle-down")
        elif token in _FLAT_TOKENS:
            styles[cls] = LabelStyle(FLAT_COLOR, "circle", 0.7)
        else:
            styles[cls] = LabelStyle(
                PALETTE[spare % len(PALETTE)], SYMBOLS[spare % len(SYMBOLS)]
            )
            spare += 1
    return styles


def compute_rangebreaks(timestamps: pd.Series) -> list[dict]:
    """Rangebreaks that collapse weekends and the overnight session.

    Without these, an intraday NIFTY futures chart wastes ~70% of its width on
    the flat stretch between 15:30 and 09:15 the next morning.
    """
    ts = pd.Series(timestamps).dropna().sort_values()
    if len(ts) < 3:
        return []

    interval = pd.Series(ts).diff().dropna()
    interval = interval[interval > pd.Timedelta(0)]
    if interval.empty:
        return []
    step = interval.mode().iloc[0]

    breaks: list[dict] = []
    weekdays = ts.dt.dayofweek
    if weekdays.max() <= 4 and ts.max() - ts.min() > pd.Timedelta(days=4):
        breaks.append({"bounds": ["sat", "mon"]})

    if step >= pd.Timedelta(days=1):
        return breaks

    hours = ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600
    session_open = float(hours.min())
    session_close = float(hours.max()) + step.total_seconds() / 3600
    if session_open > 0.01 or session_close < 23.99:
        breaks.append(
            {"bounds": [round(min(session_close, 24.0), 4), round(session_open, 4)],
             "pattern": "hour"}
        )
    return breaks


@dataclass
class PanelSpec:
    """One feature sub-chart under the price chart."""

    features: list[str] = field(default_factory=list)
    normalise: str = "none"
    chart_type: str = "line"          # line | area | bar | step
    title: str = ""
    height: float = 1.0
    mark_events: bool = False
    show_zero_line: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.features


@dataclass
class ChartOptions:
    """Display switches shared by the price chart and its panels."""

    show_target: bool = True
    show_predicted: bool = True
    show_mismatch: bool = False
    target_classes: Optional[Sequence[str]] = None
    predicted_classes: Optional[Sequence[str]] = None
    target_mode: str = "below bar"
    predicted_mode: str = "above bar"
    marker_size: int = 7
    marker_pad_pct: float = 1.5
    show_volume: bool = False
    collapse_gaps: bool = True
    price_height: float = 2.2
    price_height_px: Optional[int] = None      # explicit override for the price row
    template: str = TEMPLATE_NAME
    unified_hover: bool = True
    cross_subplot_hover: bool = True
    show_spikes: bool = True
    row_height_px: int = 190
    price_style: str = "candlestick"   # candlestick | ohlc bars | close line


def _price_pad(price: pd.DataFrame, pct: float) -> float:
    if price.empty:
        return 0.0
    span = float(price["high"].max() - price["low"].min())
    if not np.isfinite(span) or span <= 0:
        span = float(abs(price["close"].iloc[0])) or 1.0
    return span * pct / 100.0


def marker_positions(price: pd.DataFrame, mode: str, pad: float) -> pd.Series:
    """Y coordinate for a label marker on each price bar."""
    mode = (mode or "below bar").lower()
    if mode == "below bar":
        return price["low"] - pad
    if mode == "above bar":
        return price["high"] + pad
    if mode == "at high":
        return price["high"]
    if mode == "at low":
        return price["low"]
    if mode == "at close":
        return price["close"]
    raise ValueError(f"Unknown marker mode: {mode!r}")


def attach_labels(price: pd.DataFrame, labels: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Left-join labels onto price bars, adding empty columns when absent."""
    out = price.copy()
    if labels is None or labels.empty:
        out["target"] = pd.Series(pd.NA, index=out.index, dtype="string")
        out["predicted"] = pd.Series(pd.NA, index=out.index, dtype="string")
        return out

    keep = [c for c in ("timestamp", "target", "predicted") if c in labels.columns]
    merged = out.merge(labels[keep], on="timestamp", how="left")
    for col in ("target", "predicted"):
        if col not in merged.columns:
            merged[col] = pd.Series(pd.NA, index=merged.index, dtype="string")
    return merged


def _label_traces(
    bars: pd.DataFrame,
    source: str,
    mode: str,
    pad: float,
    styles: dict[str, LabelStyle],
    options: ChartOptions,
    allowed: Optional[Sequence[str]],
) -> list[go.Scatter]:
    """One marker trace per label class, so each is independently toggleable."""
    if source not in bars.columns:
        return []

    present = bars[source].dropna().astype(str).unique().tolist()
    classes = [c for c in present if allowed is None or c in set(allowed)]
    ordered = [c for c in styles if c in classes] + [c for c in classes if c not in styles]

    is_prediction = source == "predicted"
    y_all = marker_positions(bars, mode, pad)
    traces: list[go.Scatter] = []

    for cls in ordered:
        rows = bars[source].astype(str) == cls
        if not rows.any():
            continue
        style = styles.get(cls, LabelStyle(PALETTE[0], "circle"))
        symbol = f"{style.symbol}-open" if is_prediction else style.symbol
        subset = bars.loc[rows]
        traces.append(
            go.Scatter(
                x=subset["timestamp"],
                y=y_all.loc[rows],
                mode="markers",
                name=f"{'Pred' if is_prediction else 'Target'}: {cls}",
                legendgroup=source,
                legendgrouptitle_text="Predicted" if is_prediction else "Target",
                marker={
                    "symbol": symbol,
                    "size": max(4, round(options.marker_size * style.scale)),
                    "color": style.color,
                    "line": {"width": 1.6 if is_prediction else 0.8, "color": style.color},
                    "opacity": 0.95,
                },
                customdata=np.column_stack(
                    [
                        np.full(len(subset), cls, dtype=object),
                        subset["open"].to_numpy(),
                        subset["high"].to_numpy(),
                        subset["low"].to_numpy(),
                        subset["close"].to_numpy(),
                    ]
                ),
                hovertemplate=(
                    f"<b>{'Predicted' if is_prediction else 'Target'}: "
                    "%{customdata[0]}</b><br>"
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    "O %{customdata[1]:.2f} · H %{customdata[2]:.2f}<br>"
                    "L %{customdata[3]:.2f} · C %{customdata[4]:.2f}"
                    "<extra></extra>"
                ),
            )
        )
    return traces


def _mismatch_trace(bars: pd.DataFrame, pad: float, options: ChartOptions) -> Optional[go.Scatter]:
    if "target" not in bars.columns or "predicted" not in bars.columns:
        return None
    both = bars["target"].notna() & bars["predicted"].notna()
    differ = both & (bars["target"].astype("string") != bars["predicted"].astype("string"))
    if not differ.any():
        return None

    subset = bars.loc[differ]
    return go.Scatter(
        x=subset["timestamp"],
        y=subset["high"] + pad * 2.2,
        mode="markers",
        name=f"Mismatch ({int(differ.sum())})",
        legendgroup="mismatch",
        marker={
            "symbol": "x-thin",
            "size": options.marker_size + 2,
            "color": MISMATCH_COLOR,
            "line": {"width": 2.4, "color": MISMATCH_COLOR},
        },
        customdata=np.column_stack(
            [subset["target"].astype(str).to_numpy(), subset["predicted"].astype(str).to_numpy()]
        ),
        hovertemplate=(
            "<b>Mismatch</b><br>%{x|%Y-%m-%d %H:%M}<br>"
            "target %{customdata[0]} → predicted %{customdata[1]}"
            "<extra></extra>"
        ),
    )


def _price_traces(price: pd.DataFrame, style: str) -> list[go.BaseTraceType]:
    if style == "close line":
        return [
            go.Scatter(
                x=price["timestamp"],
                y=price["close"],
                mode="lines",
                name="Close",
                line={"width": 2, "color": INK_SOFT},
                hovertemplate="Close %{y:.2f}<extra></extra>",
            )
        ]
    # Solid-filled bodies with warm green/red, thin wicks, no whisker caps --
    # matches a trading-terminal look ("plain candlesticks") rather than the
    # engineering-diagram default.
    kwargs = dict(
        x=price["timestamp"],
        open=price["open"], high=price["high"],
        low=price["low"], close=price["close"],
        name="Price",
        increasing_line_color=CANDLE_BULL,
        decreasing_line_color=CANDLE_BEAR,
        line={"width": 1.0},
        showlegend=True,
    )
    if style == "ohlc bars":
        return [go.Ohlc(**kwargs)]
    # `_fillcolor` and `whiskerwidth` are Candlestick-only properties. Setting
    # fill and line to the same colour gives a solid body -- otherwise you get
    # an outline with a slightly translucent fill.
    return [go.Candlestick(
        **kwargs,
        increasing_fillcolor=CANDLE_BULL,
        decreasing_fillcolor=CANDLE_BEAR,
        whiskerwidth=0,
    )]


def _volume_trace(price: pd.DataFrame) -> Optional[go.Bar]:
    if "volume" not in price.columns or price["volume"].dropna().empty:
        return None
    up = price["close"] >= price["open"]
    return go.Bar(
        x=price["timestamp"],
        y=price["volume"],
        name="Volume",
        marker_color=np.where(up, BULL_COLOR, BEAR_COLOR),
        marker_line_width=0,
        opacity=0.55,
        hovertemplate="Volume %{y:,.0f}<extra></extra>",
    )


# The dtypes nifty-pipeline's na_policy treats as categories. Matching that list
# keeps the chart's idea of "category" identical to the trainer's -- notably bool,
# which pandas reports as numeric but the pipeline encodes as True/False/MISSING.
CATEGORICAL_DTYPES = ("object", "str", "string", "category", "bool")


def is_categorical(series: pd.Series) -> bool:
    """True for text/boolean features -- anything that is not a measurement."""
    values = pd.Series(series)
    if str(values.dtype) in CATEGORICAL_DTYPES:
        return True
    return not pd.api.types.is_numeric_dtype(values)


def categorical_codes(
    series: pd.Series,
    levels: Optional[Sequence[str]] = None,
) -> tuple[pd.Series, list[str]]:
    """Map a text feature to stable integer codes plus their labels.

    Sorted so the same category always lands on the same level, whichever slice
    of the data is on screen -- otherwise a line would jump between windows for
    no reason. When ``levels`` is provided (e.g. the sorted whole-series
    domain) it is used as the lookup instead of the observed slice's uniques,
    which is what keeps a categorical panel's y-axis stable as the user pans.

    Any value in ``series`` not present in ``levels`` maps to NaN -- callers
    should treat it as "unknown category" the same way missing input is treated.
    """
    text = pd.Series(series).astype("string")
    if levels is None:
        levels = sorted(text.dropna().unique().tolist())
    else:
        levels = list(levels)
    lookup = {value: index for index, value in enumerate(levels)}
    return text.map(lookup).astype("Float64"), levels


def _categorical_trace(
    name: str, data: pd.DataFrame, style: "FeatureStyle"
) -> go.Scatter:
    """A text feature as a step line over its category levels.

    Stepped, not interpolated: a category holds until it changes, and a sloped
    line between two states would imply values that never existed.
    """
    codes, levels = categorical_codes(data[name])
    labels = pd.Series(data[name]).astype("string").fillna("—")
    return go.Scatter(
        x=data["timestamp"],
        y=codes,
        name=name,
        mode="lines",
        line={"width": 2, "color": style.color, "dash": style.dash, "shape": "hv"},
        connectgaps=False,
        customdata=labels.to_numpy(),
        hovertemplate=f"{name}: <b>%{{customdata}}</b><extra></extra>",
        meta={"categorical": True, "levels": levels},
    )


def _feature_traces(
    panel: PanelSpec,
    data: pd.DataFrame,
    styles: dict[str, "FeatureStyle"],
) -> list[go.BaseTraceType]:
    """Build one trace per selected feature, honouring the panel's normalisation."""
    traces: list[go.BaseTraceType] = []
    normalised = panel.normalise not in ("none", "", None)

    for name in panel.features:
        if name not in data.columns:
            continue
        style = styles.get(name) or FeatureStyle(PALETTE[0], DASHES[0])

        # Text features have no magnitude, so normalisation would be meaningless.
        if is_categorical(data[name]):
            traces.append(_categorical_trace(name, data, style))
            continue

        raw = pd.to_numeric(data[name], errors="coerce")
        values = transforms.normalise_series(raw, panel.normalise)
        color = style.color

        if normalised:
            customdata = raw.to_numpy()
            hover = f"{name}: %{{y:.4f}} <i>(raw %{{customdata:.4f}})</i><extra></extra>"
        else:
            customdata = None
            hover = f"{name}: %{{y:.4f}}<extra></extra>"

        if panel.chart_type == "bar":
            traces.append(
                go.Bar(
                    x=data["timestamp"], y=values, name=name, marker_color=color,
                    marker_line_width=0, opacity=0.75,
                    customdata=customdata, hovertemplate=hover,
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=data["timestamp"],
                    y=values,
                    name=name,
                    mode="lines",
                    line={
                        "width": 2,
                        "color": color,
                        "dash": style.dash,
                        "shape": "hv" if panel.chart_type == "step" else "linear",
                    },
                    fill="tozeroy" if panel.chart_type == "area" else None,
                    connectgaps=False,
                    customdata=customdata,
                    hovertemplate=hover,
                )
            )
    return traces


def _event_marker_trace(
    events: pd.Series,
    y_low: float,
    y_high: float,
    color: str,
    name: str,
) -> Optional[go.Scatter]:
    """All label timestamps as one trace of vertical segments (cheap for 1000s of events)."""
    times = pd.Series(events).dropna()
    if times.empty or not np.isfinite(y_low) or not np.isfinite(y_high):
        return None
    xs: list = []
    ys: list = []
    for t in times:
        xs.extend([t, t, None])
        ys.extend([y_low, y_high, None])
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        name=name,
        line={"width": 1, "color": color, "dash": "dot"},
        opacity=0.5,
        hoverinfo="skip",
        showlegend=True,
        legendgroup="events",
    )


@dataclass(frozen=True)
class FeatureStyle:
    color: str
    dash: str


def feature_style_map(names: Sequence[str]) -> dict[str, FeatureStyle]:
    """Stable feature -> (colour, dash) assignment, shared by every panel.

    Colours are handed out by slot in fixed order. Once the eight hues are used
    up the dash pattern advances, so a ninth series is distinguished by colour
    *and* line style rather than by an invented hue.
    """
    return {
        name: FeatureStyle(
            PALETTE[i % len(PALETTE)],
            DASHES[(i // len(PALETTE)) % len(DASHES)],
        )
        for i, name in enumerate(names)
    }


def feature_color_map(names: Sequence[str]) -> dict[str, str]:
    """Colour-only view of :func:`feature_style_map`."""
    return {name: style.color for name, style in feature_style_map(names).items()}


def _coerce_styles(styles: Optional[dict]) -> dict[str, FeatureStyle]:
    """Accept either ``{name: FeatureStyle}`` or a plain ``{name: "#hex"}`` map."""
    if not styles:
        return {}
    return {
        name: value if isinstance(value, FeatureStyle) else FeatureStyle(str(value), DASHES[0])
        for name, value in styles.items()
    }


def build_figure(
    price: pd.DataFrame,
    labels: Optional[pd.DataFrame] = None,
    features: Optional[pd.DataFrame] = None,
    panels: Sequence[PanelSpec] = (),
    options: Optional[ChartOptions] = None,
    styles: Optional[dict[str, FeatureStyle]] = None,
) -> go.Figure:
    """Assemble the full stacked figure.

    Rows: price, [volume], then one row per entry in ``panels``. ``features``
    must already be aligned to ``price`` on ``timestamp``.
    """
    options = options or ChartOptions()
    panels = list(panels)
    if price is None or price.empty:
        raise ValueError("Price data is required to build the chart.")

    show_volume = options.show_volume and "volume" in price.columns
    row_heights = [options.price_height]
    titles = [""]
    if show_volume:
        row_heights.append(0.55)
        titles.append("")
    for panel in panels:
        row_heights.append(max(panel.height, 0.3))
        titles.append(panel.title or "")

    fig = make_subplots(
        rows=len(row_heights),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035 if len(row_heights) > 1 else 0.0,
        row_heights=row_heights,
        subplot_titles=titles,
    )

    bars = attach_labels(price, labels)
    pad = _price_pad(price, options.marker_pad_pct)

    for trace in _price_traces(price, options.price_style):
        fig.add_trace(trace, row=1, col=1)

    classes = sorted(
        set(bars["target"].dropna().astype(str)) | set(bars["predicted"].dropna().astype(str))
    )
    # Distinct from the `styles` parameter, which carries *feature* styling.
    label_styles = style_for_labels(classes)

    if options.show_target:
        for trace in _label_traces(
            bars, "target", options.target_mode, pad, label_styles, options,
            options.target_classes,
        ):
            fig.add_trace(trace, row=1, col=1)
    if options.show_predicted:
        for trace in _label_traces(
            bars, "predicted", options.predicted_mode, pad, label_styles, options,
            options.predicted_classes,
        ):
            fig.add_trace(trace, row=1, col=1)
    if options.show_mismatch:
        mismatch = _mismatch_trace(bars, pad, options)
        if mismatch is not None:
            fig.add_trace(mismatch, row=1, col=1)

    next_row = 2
    if show_volume:
        vol = _volume_trace(price)
        if vol is not None:
            fig.add_trace(vol, row=next_row, col=1)
        fig.update_yaxes(title_text="Vol", row=next_row, col=1)
        next_row += 1

    styles = _coerce_styles(styles) or feature_style_map(
        sorted({f for panel in panels for f in panel.features})
    )
    aligned = features if features is not None else pd.DataFrame({"timestamp": price["timestamp"]})

    for panel in panels:
        row = next_row
        next_row += 1
        for trace in _feature_traces(panel, aligned, styles):
            fig.add_trace(trace, row=row, col=1)

        if panel.show_zero_line:
            fig.add_hline(y=0, line_width=1, line_dash="dot",
                          line_color=INK_MUTED, row=row, col=1)

        if panel.mark_events and not panel.is_empty:
            present = [f for f in panel.features if f in aligned.columns]
            if present:
                stacked = pd.concat(
                    [transforms.normalise_series(aligned[f], panel.normalise) for f in present]
                )
                low, high = stacked.min(), stacked.max()
                marks = bars.loc[bars["target"].notna(), "timestamp"]
                trace = _event_marker_trace(marks, low, high, FLAT_COLOR, "Target events")
                if trace is not None:
                    fig.add_trace(trace, row=row, col=1)

        # A panel showing one text feature gets its categories as tick labels,
        # so the levels are readable without hovering.
        present = [f for f in panel.features if f in aligned.columns]
        categoricals = [f for f in present if is_categorical(aligned[f])]
        if len(present) == 1 and categoricals:
            _, levels = categorical_codes(aligned[present[0]])
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(len(levels))),
                ticktext=[str(v)[:22] for v in levels],
                range=[-0.5, max(len(levels) - 0.5, 0.5)],
                row=row, col=1,
            )

        label = panel.title or ("Features" if not panel.is_empty else "")
        if panel.normalise not in ("none", "", None) and len(categoricals) < len(present):
            label = f"{label} ({panel.normalise})".strip()
        fig.update_yaxes(title_text=label[:28], row=row, col=1)

    _apply_layout(fig, price, options, n_rows=len(row_heights))
    return fig


def _apply_layout(fig: go.Figure, price: pd.DataFrame, options: ChartOptions, n_rows: int) -> None:
    price_px = options.price_height_px or int(options.price_height * options.row_height_px)
    height = price_px + (n_rows - 1) * options.row_height_px
    fig.update_layout(
        template=options.template,
        height=max(height, 420),
        margin={"l": 60, "r": 25, "t": 40, "b": 40},
        hovermode="x unified" if options.unified_hover else "closest",
        dragmode="pan",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.015,
            "xanchor": "left",
            "x": 0,
            "groupclick": "toggleitem",
        },
        barmode="overlay",
        uirevision="visualizer",
    )
    # Streamlit injects its own explicit paper/plot colours into every figure,
    # which would beat the template. Restate ours so the chart renders on the
    # surface the palette was actually validated against.
    if options.template == TEMPLATE_NAME:
        fig.update_layout(
            paper_bgcolor=PLANE, plot_bgcolor=SURFACE, font_color=INK_SOFT
        )
    if options.cross_subplot_hover and n_rows > 1:
        # Hover once, read every panel at that timestamp.
        fig.update_layout(hoversubplots="axis")

    fig.update_xaxes(showgrid=True, gridcolor=GRID, zerolinecolor=AXIS_LINE)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zerolinecolor=AXIS_LINE, fixedrange=False)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    # Whole numbers with a thousands separator (24,860 not 24.86k). The hover
    # keeps two decimals for tick-precision. `autorangeoptions.include=[low, high]`
    # locks the y-range to the actual data span (plus label headroom) so the
    # candles fill the row instead of floating in a mostly-empty axis.
    pad = float(price["high"].max() - price["low"].min()) * 0.05
    fig.update_yaxes(
        title_text="Price",
        tickformat=",d",
        hoverformat=",.2f",
        autorange=True,
        autorangeoptions={
            "include": [float(price["low"].min()) - pad,
                        float(price["high"].max()) + pad],
        },
        row=1, col=1,
    )

    if options.show_spikes:
        fig.update_xaxes(
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikethickness=1, spikedash="dot", spikecolor=INK_MUTED,
        )

    if options.collapse_gaps:
        breaks = compute_rangebreaks(price["timestamp"])
        if breaks:
            fig.update_xaxes(rangebreaks=breaks)

    for annotation in fig.layout.annotations:
        annotation.font.size = 12
        annotation.font.color = INK_MUTED


def build_confusion_heatmap(matrix: pd.DataFrame, normalised: bool = False) -> go.Figure:
    """Target-vs-predicted confusion matrix as a heatmap."""
    text_fmt = "{:.0%}" if normalised else "{:,.0f}"
    text = [[text_fmt.format(v) for v in row] for row in matrix.to_numpy()]
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=[str(c) for c in matrix.columns],
            y=[str(i) for i in matrix.index],
            text=text,
            texttemplate="%{text}",
            textfont={"color": INK},
            colorscale=[list(stop) for stop in SEQUENTIAL],
            showscale=False,
            xgap=2, ygap=2,                       # surface gap between cells
            hovertemplate="target %{y} → predicted %{x}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        height=90 + 46 * max(len(matrix.index), 1),
        margin={"l": 60, "r": 20, "t": 30, "b": 40},
        xaxis_title="predicted",
        yaxis_title="target",
        yaxis_autorange="reversed",
        template=TEMPLATE_NAME,
        paper_bgcolor=PLANE,
        plot_bgcolor=SURFACE,
        font_color=INK_SOFT,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig
