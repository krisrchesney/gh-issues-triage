# ruff: noqa: E501  (SVG and CSS templates are easier to read unwrapped)
"""Charts for the static site: curve math plus small, dependency-free inline SVG renderers.

Colors are CSS custom properties (--s1.. series, --ink.. chrome) defined once in the page, so the light
and dark steps swap in one place. Series palette: blue, orange, aqua, yellow from the documented categorical
order, validated (all pairs) against this page's light (#fbfbfa) and dark (#151513) surfaces. Aqua is
below 3:1 on light, so every line carries a direct label and every chart has a table view.
"""

from __future__ import annotations

import html
import json

import numpy as np

CHART_CSS = """
:root { --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --grid:#e3e2dc; --axis:#c3c2b7; --ink-muted:#898781; --surface:#fbfbfa; }
@media (prefers-color-scheme: dark) { :root { --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --grid:#2c2c2a; --axis:#383835; --surface:#151513; } }
.chart { margin:12px 0 4px; overflow-x:auto; }
.chart svg { width:100%; min-width:330px; max-width:620px; height:auto; display:block; font:13px system-ui,-apple-system,"Segoe UI",sans-serif; }
.chart svg.square { max-width:520px; }
.chart .grid line { stroke:var(--grid); stroke-width:1; }
.chart .axis { stroke:var(--axis); stroke-width:1; }
.chart .tick { fill:var(--ink-muted); font-variant-numeric:tabular-nums; }
.chart .lbl { fill:var(--fg); }
.chart .muted { fill:var(--ink-muted); }
.chart .ref { stroke:var(--ink-muted); stroke-width:1; }
.chart .series { fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }
.chart .dot { stroke:var(--surface); stroke-width:2; }
.chart .hit { fill:transparent; cursor:crosshair; }
.chart .hit:focus { outline:none; }
.chart .xhair { stroke:var(--ink-muted); stroke-width:1; visibility:hidden; }
.legend { display:flex; flex-wrap:wrap; gap:6px 16px; font-size:.85rem; color:var(--muted); margin:4px 0; }
.legend span::before { content:""; display:inline-block; width:14px; height:2px; vertical-align:middle; margin-right:6px; background:var(--k); }
.legend .dotkey::before { width:8px; height:8px; border-radius:50%; }
#tip { position:fixed; pointer-events:none; z-index:10; background:var(--bg); color:var(--fg); border:1px solid var(--line);
  border-radius:6px; padding:6px 9px; font-size:.82rem; line-height:1.4; box-shadow:0 2px 8px rgba(0,0,0,.12); display:none; max-width:260px; }
#tip .row { display:flex; align-items:center; gap:6px; white-space:nowrap; }
#tip .row i { display:inline-block; width:12px; height:2px; }
#tip b { font-variant-numeric:tabular-nums; }
details.tableview { margin:4px 0 8px; font-size:.88rem; }
details.tableview table { min-width:0; width:auto; }
"""

# One tooltip for the whole page. Values come from JSON or data-* attributes and are inserted with
# textContent only. Line charts get a crosshair that snaps to the nearest x; scatter dots have their own
# 24px hit areas. Keyboard: hit areas are focusable and show the same readout.
CHART_JS = """
(() => {
  const tip = document.getElementById('tip');
  const show = (ev, rows, title) => {
    tip.replaceChildren();
    if (title) { const t = document.createElement('div'); t.textContent = title; t.style.color = 'var(--muted)'; tip.append(t); }
    for (const [color, value, name] of rows) {
      const r = document.createElement('div'); r.className = 'row';
      if (color) { const k = document.createElement('i'); k.style.background = color; r.append(k); }
      const b = document.createElement('b'); b.textContent = value; r.append(b);
      const s = document.createElement('span'); s.textContent = name; r.append(s);
      tip.append(r);
    }
    tip.style.display = 'block';
    const box = ev.target.getBoundingClientRect();
    const x = ev.clientX ?? (box.left + box.width / 2), y = ev.clientY ?? box.top;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(window.innerWidth - w - 8, x + 14) + 'px';
    tip.style.top = Math.max(8, y - h - 10) + 'px';
  };
  const hide = () => { tip.style.display = 'none'; };

  for (const svg of document.querySelectorAll('svg[data-line]')) {
    const d = JSON.parse(svg.dataset.line);
    const hit = svg.querySelector('.hit'), xh = svg.querySelector('.xhair');
    const at = (clientX) => {
      const pt = svg.createSVGPoint(); pt.x = clientX; pt.y = 0;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      const fx = (p.x - d.x0) / (d.x1 - d.x0) * d.xmax;
      let i = 0, best = Infinity;
      d.grid.forEach((g, k) => { const e = Math.abs(g - fx); if (e < best) { best = e; i = k; } });
      return i;
    };
    const render = (ev, i) => {
      const x = d.x0 + d.grid[i] / d.xmax * (d.x1 - d.x0);
      xh.setAttribute('x1', x); xh.setAttribute('x2', x); xh.style.visibility = 'visible';
      const fmt = v => v == null ? 'n/a' : d.fmt === 'num' ? v.toFixed(3) : (v * 100).toFixed(1) + '%';
      show(ev, d.series.map(s => [s.color, fmt(s.y[i]), s.name]),
           d.xlabels ? d.xlabels[i] : 'Top ' + (d.grid[i] * 100).toFixed(1) + '% of the list reviewed');
    };
    hit.addEventListener('pointermove', ev => render(ev, at(ev.clientX)));
    hit.addEventListener('pointerleave', () => { hide(); xh.style.visibility = 'hidden'; });
    hit.addEventListener('focus', ev => render(ev, d.grid.findIndex(g => g >= 0.05)));
    hit.addEventListener('blur', () => { hide(); xh.style.visibility = 'hidden'; });
  }
  for (const el of document.querySelectorAll('[data-tip]')) {
    const rows = JSON.parse(el.dataset.tip);
    el.addEventListener('pointerenter', ev => show(ev, rows.rows, rows.title));
    el.addEventListener('pointermove', ev => show(ev, rows.rows, rows.title));
    el.addEventListener('pointerleave', hide);
    el.addEventListener('focus', ev => show(ev, rows.rows, rows.title));
    el.addEventListener('blur', hide);
  }
})();
"""


def catch_curve(scores: np.ndarray, labels: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Share of positives found when the top `grid` fraction of items (by score) is reviewed.

    Tied scores are reviewed as a block, with positives spread evenly across it (the expected result of
    breaking ties at random). This matters for EPSS, which gives many new CVEs the same low score.
    Missing scores rank last.
    """
    s = np.where(np.isnan(scores), -np.inf, scores.astype(float))
    y = labels.astype(float)
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    # End of each tie block: cumulative items and positives there.
    ends = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    xs = np.r_[0.0, (ends + 1) / len(s)]
    ys = np.r_[0.0, np.cumsum(y)[ends] / max(y.sum(), 1)]
    return np.interp(grid, xs, ys)


def line_chart(chart_id: str, title: str, grid: np.ndarray, series: list[dict], xmax: float = 0.25) -> str:
    """series: [{"name", "var" (css var name, e.g. "--s1"), "y" (array matching grid)}]. Adds a random line."""
    W, H, L, R, T, B = 500, 300, 42, 92, 14, 40
    x0, x1, y0, y1 = L, W - R, H - B, T
    keep = grid <= xmax + 1e-12
    g = grid[keep]
    X = lambda v: x0 + v / xmax * (x1 - x0)  # noqa: E731
    Y = lambda v: y0 - v * (y0 - y1)  # noqa: E731
    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="{chart_id}-t" data-line="__DATA__">',
        f'<title id="{chart_id}-t">{html.escape(title)}</title>',
        '<g class="grid">',
    ]
    for v in (0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
    parts.append("</g>")
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<text class="tick" x="{x0 - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v * 100:.0f}%</text>'
        )
    for v in np.linspace(0, xmax, 6):
        parts.append(
            f'<text class="tick" x="{X(v):.1f}" y="{y0 + 16}" text-anchor="middle">{v * 100:.0f}%</text>'
        )
    parts.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}"/>')
    parts.append(
        f'<text class="muted" x="{(x0 + x1) / 2:.0f}" y="{H - 6}" text-anchor="middle">share of the list reviewed, highest-ranked first</text>'
    )
    # Random review: catches positives in proportion to the share reviewed.
    parts.append(f'<line class="ref" x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(xmax):.1f}" y2="{Y(xmax):.1f}"/>')
    parts.append(f'<text class="muted" x="{X(xmax) + 8:.1f}" y="{Y(xmax) + 4:.1f}">random</text>')
    # Direct end labels: keep them only where they don't collide (min 14px apart); the legend always carries identity.
    ends = sorted(((Y(float(np.asarray(s["y"])[keep][-1])), s) for s in series), key=lambda t: t[0])
    labelled, last = set(), -1e9
    for ypos, s in ends:
        if ypos - last >= 14:
            labelled.add(s["name"])
            last = ypos
    for s in series:
        ys = np.asarray(s["y"])[keep]
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(g, ys, strict=True))
        parts.append(f'<polyline class="series" style="stroke:var({s["var"]})" points="{pts}"/>')
        parts.append(
            f'<circle class="dot" r="4" cx="{X(g[-1]):.1f}" cy="{Y(ys[-1]):.1f}" style="fill:var({s["var"]})"/>'
        )
        if s["name"] in labelled:
            parts.append(
                f'<text class="lbl" x="{X(g[-1]) + 8:.1f}" y="{Y(ys[-1]) + 4:.1f}">{ys[-1] * 100:.0f}% {html.escape(s.get("short", s["name"]))}</text>'
            )
    parts.append(f'<line class="xhair" x1="0" x2="0" y1="{y1}" y2="{y0}"/>')
    parts.append(
        f'<rect class="hit" tabindex="0" x="{x0}" y="{y1}" width="{x1 - x0}" height="{y0 - y1}"><title>{html.escape(title)}</title></rect>'
    )
    parts.append("</svg>")
    data = {
        "x0": x0,
        "x1": x1,
        "xmax": xmax,
        "grid": [round(float(v), 5) for v in g],
        "series": [
            {
                "name": s["name"],
                "color": f"var({s['var']})",
                "y": [round(float(v), 4) for v in np.asarray(s["y"])[keep]],
            }
            for s in series
        ],
    }
    svg = "".join(parts).replace("__DATA__", html.escape(json.dumps(data), quote=True))
    legend = (
        '<div class="legend">'
        + "".join(f'<span style="--k:var({s["var"]})">{html.escape(s["name"])}</span>' for s in series)
        + '<span style="--k:var(--ink-muted)">random review</span></div>'
    )
    rows = "".join(
        "<tr><td>"
        + html.escape(s["name"])
        + "</td>"
        + "".join(
            f'<td class="num">{float(np.interp(p, grid, s["y"])) * 100:.1f}%</td>'
            for p in (0.01, 0.02, 0.05, 0.10, 0.20)
        )
        + "</tr>"
        for s in series
    )
    table = (
        '<details class="tableview"><summary>Show as a table</summary><div class="wrap"><table><thead><tr><th>Ranking</th>'
        + "".join(f'<th class="num">top {p}%</th>' for p in (1, 2, 5, 10, 20))
        + f"</tr></thead><tbody>{rows}</tbody></table></div></details>"
    )
    return f'{legend}<div class="chart">{svg}</div>{table}'


def scatter_chart(chart_id: str, title: str, points: list[dict], xlabel: str, ylabel: str) -> str:
    """points: [{"x","y" (0..1 percentiles), "tip": {"title", "rows": [[color, value, name], ...]}}]."""
    W, H, L, R, T, B = 440, 400, 44, 14, 14, 42
    x0, x1, y0, y1 = L, W - R, H - B, T
    pad = 8  # keeps dots at 0 or 100 clear of the axes
    X = lambda v: x0 + pad + v * (x1 - x0 - 2 * pad)  # noqa: E731
    Y = lambda v: y0 - pad - v * (y0 - y1 - 2 * pad)  # noqa: E731
    parts = [
        f'<svg class="square" viewBox="0 0 {W} {H}" role="img" aria-labelledby="{chart_id}-t">',
        f'<title id="{chart_id}-t">{html.escape(title)}</title>',
        '<g class="grid">',
    ]
    for v in (0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        parts.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{y1}" y2="{y0}"/>')
    parts.append("</g>")
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<text class="tick" x="{x0 - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v * 100:.0f}</text>'
        )
        parts.append(
            f'<text class="tick" x="{X(v):.1f}" y="{y0 + 16}" text-anchor="middle">{v * 100:.0f}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}"/><line class="axis" x1="{x0}" x2="{x0}" y1="{y1}" y2="{y0}"/>'
    )
    parts.append(f'<line class="ref" x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(1):.1f}" y2="{Y(1):.1f}"/>')
    # No in-plot annotations: points move daily and any fixed label would collide with them. The caption
    # explains the diagonal.
    parts.append(
        f'<text class="muted" x="{(x0 + x1) / 2:.0f}" y="{H - 6}" text-anchor="middle">{html.escape(xlabel)}</text>'
    )
    parts.append(
        f'<text class="muted" transform="translate(12 {(y0 + y1) / 2:.0f}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>'
    )
    for p in points:
        cx, cy = X(p["x"]), Y(p["y"])
        tip = html.escape(json.dumps(p["tip"]), quote=True)
        parts.append(f'<circle class="dot" r="4" cx="{cx:.1f}" cy="{cy:.1f}" style="fill:var(--s1)"/>')
        parts.append(
            f'<circle class="hit" tabindex="0" r="12" cx="{cx:.1f}" cy="{cy:.1f}" data-tip="{tip}"/>'
        )
    parts.append("</svg>")
    return f'<div class="chart">{"".join(parts)}</div>'


def series_chart(
    chart_id: str,
    title: str,
    xlabels: list[str],
    series: list[dict],
    y_lo: float,
    y_hi: float,
    fmt: str = "pct",
    ref: float | None = None,
    ref_label: str = "",
    xlabel: str = "",
    tick_step: float = 0.25,
) -> str:
    """Lines over evenly spaced categories (e.g. weeks). series: [{"name", "short", "var", "y": list with
    None for missing}]. fmt "pct" shows 0.42 as 42%, "num" as 0.420. Optional horizontal reference line."""
    W, H, L, R, T, B = 500, 300, 42, 112, 14, 40
    x0, x1, y0, y1 = L, W - R, H - B, T
    n = len(xlabels)
    X = lambda i: x0 + (i / max(n - 1, 1)) * (x1 - x0)  # noqa: E731
    Y = lambda v: y0 - (v - y_lo) / (y_hi - y_lo) * (y0 - y1)  # noqa: E731
    label = (
        (lambda v: f"{v * 100:.0f}%")
        if fmt == "pct"
        else (lambda v: f"{v:.1f}" if abs(v * 10 - round(v * 10)) < 1e-9 else f"{v:.2f}")
    )
    ticks = np.arange(y_lo, y_hi + tick_step / 2, tick_step)  # clean, evenly stepped ticks
    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="{chart_id}-t" data-line="__DATA__">',
        f'<title id="{chart_id}-t">{html.escape(title)}</title>',
        '<g class="grid">',
    ]
    for v in ticks[1:]:
        parts.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
    parts.append("</g>")
    for v in ticks:
        parts.append(
            f'<text class="tick" x="{x0 - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{label(v)}</text>'
        )
    step = max(1, int(np.ceil(n / 6)))  # at most ~6 x labels so they never collide
    for i in range(0, n, step):
        parts.append(
            f'<text class="tick" x="{X(i):.1f}" y="{y0 + 16}" text-anchor="middle">{html.escape(xlabels[i])}</text>'
        )
    parts.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}"/>')
    if xlabel:
        parts.append(
            f'<text class="muted" x="{(x0 + x1) / 2:.0f}" y="{H - 6}" text-anchor="middle">{html.escape(xlabel)}</text>'
        )
    if ref is not None:
        parts.append(f'<line class="ref" x1="{x0}" x2="{x1}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}"/>')
        parts.append(f'<text class="muted" x="{x1 + 8}" y="{Y(ref) + 4:.1f}">{html.escape(ref_label)}</text>')
    last_y = {s["name"]: next((v for v in reversed(s["y"]) if v is not None), None) for s in series}
    ends = sorted(((Y(v), s) for s in series if (v := last_y[s["name"]]) is not None), key=lambda t: t[0])
    labelled, prev = set(), -1e9
    for ypos, s in ends:
        if ypos - prev >= 14:
            labelled.add(s["name"])
            prev = ypos
    for s in series:
        seg: list[str] = []
        segments = []
        for i, v in enumerate(s["y"]):
            if v is None:
                if seg:
                    segments.append(seg)
                seg = []
            else:
                seg.append(f"{X(i):.1f},{Y(v):.1f}")
        if seg:
            segments.append(seg)
        for pts in segments:
            parts.append(
                f'<polyline class="series" style="stroke:var({s["var"]})" points="{" ".join(pts)}"/>'
            )
        for i, v in enumerate(s["y"]):
            if v is not None:
                parts.append(
                    f'<circle class="dot" r="4" cx="{X(i):.1f}" cy="{Y(v):.1f}" style="fill:var({s["var"]})"/>'
                )
        if s["name"] in labelled:
            i_last = max(i for i, v in enumerate(s["y"]) if v is not None)
            parts.append(
                f'<text class="lbl" x="{X(i_last) + 8:.1f}" y="{Y(last_y[s["name"]]) + 4:.1f}">'
                f"{label(last_y[s['name']])} {html.escape(s.get('short', s['name']))}</text>"
            )
    parts.append(f'<line class="xhair" x1="0" x2="0" y1="{y1}" y2="{y0}"/>')
    parts.append(
        f'<rect class="hit" tabindex="0" x="{x0 - 10}" y="{y1}" width="{x1 - x0 + 20}" height="{y0 - y1}"><title>{html.escape(title)}</title></rect>'
    )
    parts.append("</svg>")
    data = {
        "x0": x0,
        "x1": x1,
        "xmax": 1,
        "grid": [round(i / max(n - 1, 1), 5) for i in range(n)],
        "xlabels": xlabels,
        "fmt": fmt,
        "series": [
            {
                "name": s["name"],
                "color": f"var({s['var']})",
                "y": [None if v is None else round(float(v), 4) for v in s["y"]],
            }
            for s in series
        ],
    }
    svg = "".join(parts).replace("__DATA__", html.escape(json.dumps(data), quote=True))
    legend = (
        '<div class="legend">'
        + "".join(f'<span style="--k:var({s["var"]})">{html.escape(s["name"])}</span>' for s in series)
        + (
            f'<span style="--k:var(--ink-muted)">{html.escape(ref_label)}</span>'
            if ref is not None and ref_label
            else ""
        )
        + "</div>"
    )
    cell = (
        (lambda v: "n/a" if v is None else label(v))
        if fmt == "pct"
        else (lambda v: "n/a" if v is None else f"{v:.3f}")
    )
    rows = "".join(
        f"<tr><td>{html.escape(x)}</td>"
        + "".join(f'<td class="num">{cell(s["y"][i])}</td>' for s in series)
        + "</tr>"
        for i, x in enumerate(xlabels)
    )
    table = (
        '<details class="tableview"><summary>Show as a table</summary><div class="wrap"><table><thead><tr><th></th>'
        + "".join(f'<th class="num">{html.escape(s["name"])}</th>' for s in series)
        + f"</tr></thead><tbody>{rows}</tbody></table></div></details>"
    )
    return f'{legend}<div class="chart">{svg}</div>{table}'
