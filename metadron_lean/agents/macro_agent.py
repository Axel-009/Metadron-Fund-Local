"""Macro agent — writes the two ObjectStore files the algorithm reads.

    metadron/macro_inputs.json   numeric readings for the question tree, forward Rm and GMTF
    metadron/scenarios.json      base / up / down what-if scenarios (Claude, framework in
                                 agents/market_reasoning.md), validated by sector_rotation.scenarios_from_json

Runs OFF the trading node (owner machine, cron, or a Claude Managed Agent) once a day before the open:

    FRED_API_KEY=...  ANTHROPIC_API_KEY=...  QC_USER_ID=...  QC_API_TOKEN=...  QC_ORG_ID=...
    python3 agents/macro_agent.py [--no-llm] [--no-upload] [--out agents/out]

Every step degrades gracefully (design rule 2): no FRED key → readings omitted; no Anthropic key or
--no-llm → no scenarios file; no QC credentials or --no-upload → files are only written to --out for
manual upload. The agent is propose-only (design rule 4): it writes inputs, never orders.

Readings (all decimals; names match algorithm/macro_regime.py and forward_rm.py):
    inflation_yoy          CPIAUCSL y/y
    unemployment           UNRATE (3-month average)          unemployment_12m_low  min of last 12
    real_wage_yoy          AHETPI y/y − CPI y/y              productivity_yoy      OPHNFB y/y
    rates_change_3m        DGS10 change over ~63 business days
    liquidity_change_3m    net liquidity (WALCL − WTREGEN − RRPONTSYD) 3-month % change
    hy_oas_change_1m       BAMLH0A0HYM2 change over ~21 business days
    growth_surprise        z-score of the latest INDPRO 3-month growth vs its 5-year history
                           (proxy: FRED has no consensus expectations — agents may override)
    gmtf_score             GMTF tension percentile 0..1 (engine.signals.macro_engine.compute_gmtf)
Optional extras an agent or the owner may add to the file: earnings_yield, escalation, kronos_spx_annual.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "algorithm"))
sys.path.insert(0, ROOT)

from macro_regime import diagnose, factor_states  # noqa: E402
from sector_rotation import scenarios_from_json   # noqa: E402

DEFAULT_MODEL = "claude-opus-5-5"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
Series = List[Tuple[str, float]]               # [(date, value)] oldest → newest


# ---------------------------------------------------------------------------- FRED
def fred_fetch(series_id: str, start: str = "2015-01-01", api_key: Optional[str] = None) -> Series:
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        return []
    q = urllib.parse.urlencode({"series_id": series_id, "api_key": key, "file_type": "json",
                                "observation_start": start})
    try:
        with urllib.request.urlopen(f"{FRED_URL}?{q}", timeout=30) as r:
            obs = json.load(r).get("observations", [])
    except Exception as exc:
        print(f"[macro-agent] FRED {series_id} failed: {exc}", file=sys.stderr)
        return []
    out = []
    for o in obs:
        try:
            out.append((o["date"], float(o["value"])))
        except (KeyError, ValueError):
            continue                               # "." = missing
    return out


def _yoy(s: Series, periods: int) -> Optional[float]:
    if len(s) <= periods or s[-1 - periods][1] == 0:
        return None
    return s[-1][1] / s[-1 - periods][1] - 1


def _change(s: Series, periods: int) -> Optional[float]:
    if len(s) <= periods:
        return None
    return s[-1][1] - s[-1 - periods][1]


def _align(*series: Series) -> List[Tuple[str, List[float]]]:
    """Inner-join series on date."""
    maps = [dict(s) for s in series]
    dates = sorted(set.intersection(*(set(m) for m in maps))) if maps else []
    return [(d, [m[d] for m in maps]) for d in dates]


def compute_readings(fetch: Callable[[str], Series]) -> Dict[str, float]:
    r: Dict[str, float] = {}
    cpi = fetch("CPIAUCSL")
    if (v := _yoy(cpi, 12)) is not None:
        r["inflation_yoy"] = v
    un = fetch("UNRATE")
    if len(un) >= 12:
        r["unemployment"] = sum(x for _, x in un[-3:]) / 300
        r["unemployment_12m_low"] = min(x for _, x in un[-12:]) / 100
    wages = fetch("AHETPI")
    if (w := _yoy(wages, 12)) is not None and "inflation_yoy" in r:
        r["real_wage_yoy"] = w - r["inflation_yoy"]
    prod = fetch("OPHNFB")                          # quarterly
    if (p := _yoy(prod, 4)) is not None:
        r["productivity_yoy"] = p
    dgs10 = fetch("DGS10")
    if (c := _change(dgs10, 63)) is not None:
        r["rates_change_3m"] = c / 100
    hy = fetch("BAMLH0A0HYM2")
    if (c := _change(hy, 21)) is not None:
        r["hy_oas_change_1m"] = c / 100
    # Net liquidity ($bn) = Fed balance sheet (WALCL, $mn) − TGA (WTREGEN, $bn) − reverse repo (RRPONTSYD, $bn)
    walcl, tga, rrp = fetch("WALCL"), fetch("WTREGEN"), fetch("RRPONTSYD")
    rows = _align(walcl, tga, rrp)                  # weekly (Wednesdays)
    if len(rows) > 13:
        net = [(d, a / 1000 - b - c) for d, (a, b, c) in rows]
        if net[-14][1] > 0:
            r["liquidity_change_3m"] = net[-1][1] / net[-14][1] - 1
    ip = fetch("INDPRO")
    if len(ip) > 66:
        g3 = [ip[i][1] / ip[i - 3][1] - 1 for i in range(3, len(ip))]
        hist = g3[-63:-1]
        m = sum(hist) / len(hist)
        sd = math.sqrt(sum((x - m) ** 2 for x in hist) / (len(hist) - 1)) or 1e-9
        r["growth_surprise"] = max(-3.0, min(3.0, (g3[-1] - m) / sd))
    return r


def compute_gmtf_score(fetch_df=None) -> Optional[float]:
    """GMTF tension percentile via the engine (needs pandas + a FRED DataFrame fetcher)."""
    try:
        import pandas as pd
        from engine.signals.macro_engine import build_gmtf_inputs, compute_gmtf
    except Exception:
        return None
    if fetch_df is None:
        def fetch_df(ids, start="2005-01-01"):
            cols = {}
            for sid in ids:
                s = fred_fetch(sid, start)
                if s:
                    cols[sid] = pd.Series({pd.Timestamp(d): v for d, v in s})
            return pd.DataFrame(cols)
    try:
        inputs = build_gmtf_inputs(fetch=fetch_df)
        if not inputs:
            return None
        out = compute_gmtf(inputs["m2"], inputs["gdp"], inputs["unemp"], inputs["fx"], inputs["rates"])
        series = out.gmtf_smoothed.replace([float("inf"), float("-inf")], float("nan")).dropna()
        if len(series) < 12:
            return None
        return float((series <= series.iloc[-1]).mean())
    except Exception as exc:
        print(f"[macro-agent] GMTF failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------- scenarios (Claude)
def build_prompt(readings: Dict[str, float]) -> str:
    findings = diagnose(readings)
    factors = factor_states(readings)
    lines = [f"- {f.question} → {f.answer} ({', '.join(f.tags) or 'no tag'})" for f in findings]
    return ("Today's readings (decimals):\n" + json.dumps(readings, indent=2, sort_keys=True)
            + "\n\nFactor states [-1, +1]:\n" + json.dumps({k: round(v, 3) for k, v in factors.items()}, indent=2)
            + "\n\nQuestion tree:\n" + "\n".join(lines)
            + "\n\nWrite 2–4 scenarios (base / up / down) for the next 1–3 months following the framework. "
              "Reply with ONLY the JSON object in the scenario schema (source \"macro-agent\", as_of today).")


def generate_scenarios(readings: Dict[str, float], model: str = DEFAULT_MODEL) -> Optional[str]:
    try:
        import anthropic
    except Exception:
        print("[macro-agent] anthropic SDK missing — no scenarios", file=sys.stderr)
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    with open(os.path.join(HERE, "market_reasoning.md"), encoding="utf-8") as fh:
        framework = fh.read()
    try:
        msg = anthropic.Anthropic().messages.create(
            model=model, max_tokens=4000, system=framework,
            messages=[{"role": "user", "content": build_prompt(readings)}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception as exc:
        print(f"[macro-agent] Claude call failed: {exc}", file=sys.stderr)
        return None
    return validate_scenarios(text)


def validate_scenarios(text: str) -> Optional[str]:
    """Extract the JSON object and check it parses into scenarios (unknown factors are dropped)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    raw = text[start:end + 1]
    try:
        obj = json.loads(raw)
        if not scenarios_from_json(raw):
            return None
    except Exception:
        return None
    obj.setdefault("source", "macro-agent")
    obj.setdefault("as_of", dt.date.today().isoformat())
    return json.dumps(obj, indent=2)


# ---------------------------------------------------------------------------- QuantConnect ObjectStore
def qc_upload(key: str, payload: str) -> bool:
    """POST to the QC REST API object/set (needs QC_USER_ID, QC_API_TOKEN, QC_ORG_ID and network access)."""
    uid, token, org = (os.environ.get(k) for k in ("QC_USER_ID", "QC_API_TOKEN", "QC_ORG_ID"))
    if not (uid and token and org):
        return False
    ts = str(int(time.time()))
    hashed = hashlib.sha256(f"{token}:{ts}".encode()).hexdigest()
    auth = base64.b64encode(f"{uid}:{hashed}".encode()).decode()
    boundary = "----metadron" + ts
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"organizationId\"\r\n\r\n{org}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"key\"\r\n\r\n{key}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"objectData\"; filename=\"data.json\"\r\n"
            f"Content-Type: application/json\r\n\r\n{payload}\r\n--{boundary}--\r\n").encode()
    req = urllib.request.Request("https://www.quantconnect.com/api/v2/object/set", data=body, method="POST",
                                 headers={"Authorization": f"Basic {auth}", "Timestamp": ts,
                                          "Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            ok = bool(json.load(r).get("success"))
    except Exception as exc:
        print(f"[macro-agent] QC upload {key} failed: {exc}", file=sys.stderr)
        return False
    return ok


# ---------------------------------------------------------------------------- main
def run(out_dir: str, use_llm: bool = True, upload: bool = True, fetch: Callable[[str], Series] = fred_fetch,
        model: str = DEFAULT_MODEL) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    readings = compute_readings(fetch)
    g = compute_gmtf_score() if os.environ.get("FRED_API_KEY") else None
    if g is not None:
        readings["gmtf_score"] = g
    readings_json = json.dumps({"as_of": dt.date.today().isoformat(), **readings}, indent=2)
    files = {"metadron/macro_inputs.json": readings_json}
    if use_llm and (scen := generate_scenarios(readings, model)):
        files["metadron/scenarios.json"] = scen
    status = {}
    for key, payload in files.items():
        path = os.path.join(out_dir, os.path.basename(key))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(payload)
        status[key] = {"file": path, "uploaded": qc_upload(key, payload) if upload else False}
    return {"readings": readings, "files": status}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    res = run(a.out, use_llm=not a.no_llm, upload=not a.no_upload, model=a.model)
    print(json.dumps(res, indent=2, default=str))
