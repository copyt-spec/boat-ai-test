# engine/txt_race_parser.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RaceMeta:
    venue: str
    date: str  # YYYYMMDD
    race_no: int
    weather: Optional[str] = None
    wind_dir: Optional[str] = None
    wind_speed_mps: Optional[float] = None
    wave_cm: Optional[float] = None


@dataclass
class BoatRow:
    finish: int
    lane: int
    racer_no: int
    motor: Optional[int] = None
    boat: Optional[int] = None
    exhibit: Optional[float] = None
    course: Optional[int] = None
    st: Optional[float] = None  # F is negative
    st_raw: Optional[str] = None


@dataclass
class RaceRecord:
    meta: RaceMeta
    boats: List[BoatRow]
    y_combo: Optional[str] = None
    trifecta_payout: Optional[int] = None


_NUM = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")

_ZEN2HAN = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "Ｒ": "R", "ｒ": "r",
    "［": "[", "］": "]",
    "－": "-", "ー": "-", "―": "-", "−": "-",
    "　": " ",
})

def _norm_line(s: str) -> str:
    s2 = s.translate(_ZEN2HAN)
    s2 = re.sub(r"[ \t]+", " ", s2)
    return s2.rstrip("\n")


def _to_float(s: str) -> Optional[float]:
    m = _NUM.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _date_to_yyyymmdd(s: str) -> Optional[str]:
    m = re.search(r"(\d{4})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})", s)
    if not m:
        return None
    y = int(m.group(1))
    mo = int(m.group(2))
    d = int(m.group(3))
    return f"{y:04d}{mo:02d}{d:02d}"


def _parse_weather_line(line: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m_weather = re.search(r"(晴れ|曇り|雨|雪|くもり)", line)
    if m_weather:
        out["weather"] = m_weather.group(1)

    m_wdir = re.search(r"(北東|南東|南西|北西|北|東|南|西)", line)
    if m_wdir:
        out["wind_dir"] = m_wdir.group(1)

    m_wspd = re.search(r"(\d+(?:\.\d+)?)\s*m", line)
    if m_wspd:
        out["wind_speed_mps"] = float(m_wspd.group(1))

    m_wave = re.search(r"(\d+(?:\.\d+)?)\s*cm", line)
    if m_wave:
        out["wave_cm"] = float(m_wave.group(1))

    return out


def _parse_st(st_token: str) -> Tuple[Optional[float], str]:
    s = st_token.strip().upper()
    if s.startswith("F"):
        v = _to_float(s)
        if v is None:
            return None, s
        return -abs(v), s
    v = _to_float(s)
    return v, s


def _split_by_kbgn_kend(all_lines: List[str]) -> List[List[str]]:
    start_re = re.compile(r"^\s*\d{2}KBGN\s*$")
    end_re = re.compile(r"^\s*\d{2}KEND\s*$")

    sections: List[List[str]] = []
    cur: List[str] = []
    in_block = False

    for raw in all_lines:
        ln = _norm_line(raw)

        if start_re.match(ln):
            cur = []
            in_block = True
            continue

        if in_block and end_re.match(ln):
            if cur:
                sections.append(cur)
            cur = []
            in_block = False
            continue

        if in_block:
            cur.append(ln)

    return sections if sections else [list(map(_norm_line, all_lines))]


def _split_by_venue_header(lines: List[str]) -> List[List[str]]:
    starts: List[int] = []
    header_re = re.compile(r"^\s*(.+?)\s*\[成績\]")

    for i, ln in enumerate(lines):
        if header_re.match(ln):
            starts.append(i)

    if not starts:
        return [lines]

    sections: List[List[str]] = []
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        sections.append(lines[s:e])
    return sections


def _extract_trifecta_map(section_lines: List[str]) -> Dict[int, Tuple[str, int]]:
    trifecta_map: Dict[int, Tuple[str, int]] = {}
    in_pay = False

    p_line = re.compile(
        r"^\s*(\d{1,2})\s*[Rr]\b"
        r".*?"
        r"([1-6])\D+([1-6])\D+([1-6])"
        r"\s+(\d{3,})\b"
    )

    p_line_kw = re.compile(
        r"^\s*(\d{1,2})\s*[Rr]\b"
        r".*?(?:3連単|３連単|三連単).*?"
        r"([1-6])\D+([1-6])\D+([1-6])"
        r"\s+(\d{3,})\b"
    )

    for raw in section_lines:
        ln = _norm_line(raw)

        if ("[払戻金]" in ln) or ("払戻金" in ln):
            in_pay = True
            continue

        if not in_pay:
            if p_line.search(ln) or p_line_kw.search(ln):
                in_pay = True
            else:
                continue

        # 本文見出しに入ったら終了（H1800mが目印）
        if re.match(r"^\s*1\s*[Rr]\b", ln) and ("H" in ln and "m" in ln):
            break

        m = p_line.search(ln) or p_line_kw.search(ln)
        if not m:
            continue

        rno = int(m.group(1))
        a, b, c = m.group(2), m.group(3), m.group(4)
        payout = int(m.group(5))
        trifecta_map[rno] = (f"{a}-{b}-{c}", payout)

    return trifecta_map


def _find_detail_race_starts(section_lines: List[str]) -> List[Tuple[int, int]]:
    """
    ✅ 本文のレース見出しだけ拾う（払戻表の 1R/2R… を除外）
    → 本文見出し行には H1800m が必ずあるので、それを条件にする
    """
    head_re = re.compile(r"^\s*(\d{1,2})\s*[Rr]\b")
    starts: List[Tuple[int, int]] = []

    for i, raw in enumerate(section_lines):
        ln = _norm_line(raw)
        m = head_re.match(ln)
        if not m:
            continue

        # ⭐ ここがキモ：H1800m（= H と m が同じ行にある）だけ採用
        if not ("H" in ln and "m" in ln):
            continue

        rno = int(m.group(1))
        if 1 <= rno <= 12:
            starts.append((rno, i))

    # インデックス順に（同じRが複数あっても最初だけ）
    starts.sort(key=lambda x: x[1])
    seen = set()
    uniq: List[Tuple[int, int]] = []
    for rno, idx in starts:
        if rno in seen:
            continue
        seen.add(rno)
        uniq.append((rno, idx))
    return uniq


def _parse_single_venue_section(section_lines: List[str], fallback_date: Optional[str]) -> List[RaceRecord]:
    lines = [_norm_line(x) for x in section_lines]

    venue = "UNKNOWN"
    header_re = re.compile(r"^\s*(.+?)\s*\[成績\]")
    for ln in lines[:80]:
        m = header_re.match(ln)
        if m:
            venue = m.group(1).strip()
            break
    if venue == "UNKNOWN":
        for ln in lines[:120]:
            if "ボートレース" in ln:
                venue = ln.replace("ボートレース", "").strip()
                break

    date = None
    for ln in lines[:300]:
        d = _date_to_yyyymmdd(ln)
        if d:
            date = d
            break
    date = date or fallback_date or "00000000"

    trifecta_map = _extract_trifecta_map(lines)
    race_starts = _find_detail_race_starts(lines)
    if not race_starts:
        return []

    row_re = re.compile(r"^\s*(\d{2})\s+([1-6])\s+(\d{4})\s+")
    records: List[RaceRecord] = []

    for idx, (rno, start_i) in enumerate(race_starts):
        end_i = race_starts[idx + 1][1] if idx + 1 < len(race_starts) else len(lines)
        block = lines[start_i:end_i]

        meta_kwargs: Dict[str, Any] = {"venue": venue, "date": date, "race_no": rno}
        for ln in block[:15]:
            if "風" in ln and "波" in ln:
                meta_kwargs.update(_parse_weather_line(ln))
                break
        meta = RaceMeta(**meta_kwargs)

        boats: List[BoatRow] = []
        for ln in block:
            m = row_re.match(ln)
            if not m:
                continue

            finish = int(m.group(1))
            lane = int(m.group(2))
            racer_no = int(m.group(3))
            tokens = ln.split()

            exhibit = None
            for t in tokens:
                if re.fullmatch(r"\d\.\d{2}", t):
                    exhibit = float(t)
                    break

            st_token = None
            for t in reversed(tokens):
                if re.fullmatch(r"F?\.\d+|F\d+\.\d+|\d+\.\d+|\.\d+", t, flags=re.IGNORECASE):
                    st_token = t
                    break
            st_val, st_raw = (None, None)
            if st_token:
                st_val, st_raw = _parse_st(st_token)

            course = None
            for t in reversed(tokens):
                if t.isdigit():
                    v = int(t)
                    if 1 <= v <= 6:
                        course = v
                        break

            motor = boat = None
            ex_idx = None
            for i_t, t in enumerate(tokens):
                if re.fullmatch(r"\d\.\d{2}", t):
                    ex_idx = i_t
                    break

            if ex_idx is not None:
                prev_nums: List[int] = []
                for t in tokens[:ex_idx]:
                    if re.fullmatch(r"\d{1,3}", t):
                        prev_nums.append(int(t))
                if len(prev_nums) >= 2:
                    motor, boat = prev_nums[-2], prev_nums[-1]

            boats.append(
                BoatRow(
                    finish=finish,
                    lane=lane,
                    racer_no=racer_no,
                    motor=motor,
                    boat=boat,
                    exhibit=exhibit,
                    course=course,
                    st=st_val,
                    st_raw=st_raw,
                )
            )

        if len(boats) < 6:
            continue

        y_combo = None
        payout = None
        if rno in trifecta_map:
            y_combo, payout = trifecta_map[rno]

        records.append(
            RaceRecord(
                meta=meta,
                boats=boats[:6],
                y_combo=y_combo,
                trifecta_payout=payout,
            )
        )

    return records


def parse_startk_multi_venue_txt(text: str, fallback_date: Optional[str] = None) -> List[RaceRecord]:
    raw_lines = text.splitlines()
    kbgn_sections = _split_by_kbgn_kend(raw_lines)

    records: List[RaceRecord] = []
    for sec in kbgn_sections:
        venue_sections = _split_by_venue_header(sec)
        for vsec in venue_sections:
            records.extend(_parse_single_venue_section(vsec, fallback_date=fallback_date))

    return records
