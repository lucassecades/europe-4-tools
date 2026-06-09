#!/usr/bin/env python3
"""
EU Portal Screening from HTML — Streamlit version
"""
import asyncio
import io
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "4. EU portal screening from HTML"))
from sedia_api import fetch_topic_data, extract_topic_id

MAX_CONCURRENT = 15

st.set_page_config(page_title="EU Portal Screening", page_icon="🌐", layout="wide")
st.title("🌐 EU Portal Screening from HTML")
st.markdown(
    "Upload HTML exports from the EU Funding & Tenders Portal, "
    "mark interesting calls, then auto-fetch descriptions via the SEDIA API."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_calls_from_html_bytes(html_bytes: bytes) -> list[dict]:
    raw = html_bytes.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    calls, seen = [], set()

    # Primary: eui-card blocks
    for card in soup.find_all("eui-card"):
        try:
            link = card.find("a", class_="eui-u-text-link")
            if not link:
                continue
            title = link.get_text(strip=True)
            url   = link.get("href", "")
            if url and not url.startswith("http"):
                url = f"https://ec.europa.eu{url}"
            call_id = None
            for span in card.find_all("span", class_="ng-star-inserted"):
                text = span.get_text(strip=True)
                if re.match(r"^(HORIZON|CASCADE|EUROSTARS|EIC|DIGITAL|LIFE|CEF|MULTIPLE)[-–][A-Z0-9\-]+", text) \
                        or (len(text) > 5 and text[:3].isupper() and "-" in text):
                    call_id = text
                    break
            if not call_id:
                m = re.search(
                    r"(HORIZON|CASCADE|EUROSTARS|EIC|DIGITAL|LIFE|CEF|MULTIPLE)[-–][A-Z0-9\-]+",
                    card.get_text(" ", strip=True), re.I
                )
                if m:
                    call_id = m.group(0)
            deadline = ""
            dl_label = card.find(string=re.compile(r"(Deadline date|Next deadline):", re.I))
            if dl_label:
                strong = dl_label.find_next("strong")
                if strong:
                    deadline = strong.get_text(strip=True)
            row = {
                "Call": call_id or title, "Title": title, "URL": url,
                "Deadline": deadline, "Interesting brubotics": "",
                "Description": "", "Budget": "", "Type": "",
            }
            key = (row["Call"], row["Title"], row["URL"])
            if key not in seen:
                seen.add(key)
                calls.append(row)
        except Exception:
            continue

    # Fallback: topic-details hrefs
    if not calls:
        for a in soup.find_all("a", href=True):
            href = a.get("href", "").strip()
            if "topic-details/" not in href:
                continue
            if not href.startswith("http"):
                href = f"https://ec.europa.eu{href}" if href.startswith("/") else f"https://ec.europa.eu/{href}"
            m = re.search(r"topic-details/([^/?#]+)", href)
            call_id = m.group(1) if m else ""
            title   = a.get_text(" ", strip=True) or call_id
            row = {
                "Call": call_id or title, "Title": title, "URL": href,
                "Deadline": "", "Interesting brubotics": "",
                "Description": "", "Budget": "", "Type": "",
            }
            key = (row["Call"], row["Title"], row["URL"])
            if key not in seen:
                seen.add(key)
                calls.append(row)

    # Last resort: regex scan
    if not calls:
        for cid in sorted(set(re.findall(
            r"\b(?:HORIZON|DIGITAL|EIC|LIFE|CEF|ERASMUS|EDF|CASCADE|EUROSTARS)[-–][A-Z0-9\-]+\b",
            raw, re.I
        ))):
            row = {
                "Call": cid, "Title": cid, "URL": "",
                "Deadline": "", "Interesting brubotics": "",
                "Description": "", "Budget": "", "Type": "",
            }
            key = (row["Call"], row["Title"], row["URL"])
            if key not in seen:
                seen.add(key)
                calls.append(row)

    return calls


async def _fill_all_async(df: pd.DataFrame, log_lines: list, progress_placeholder, log_placeholder) -> pd.DataFrame:
    for col in ("Description", "Budget", "Type"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)

    todo = [
        (idx, str(row.get("URL", "") or ""))
        for idx, row in df.iterrows()
        if str(row.get("URL", "")).strip() not in ("", "nan")
    ]

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    done = [0]

    async def fetch_one(idx, url):
        async with sem:
            data = await asyncio.to_thread(fetch_topic_data, url)
            df.at[idx, "Description"] = data["description"]
            df.at[idx, "Budget"]      = data["budget"]
            df.at[idx, "Type"]        = data["type"]
            done[0] += 1
            pct    = done[0] / len(todo) * 100
            status = "OK  " if data["description"] else "MISS"
            tid    = extract_topic_id(url) or url
            log_lines.append(f"  [{done[0]:3}/{len(todo)}  {pct:5.1f}%]  {status}  {tid}")
            log_placeholder.code("\n".join(log_lines), language=None)
            progress_placeholder.progress(done[0] / len(todo))

    await asyncio.gather(*[fetch_one(i, u) for i, u in todo])
    return df


# ── Session state ─────────────────────────────────────────────────────────────

if "portal_calls" not in st.session_state:
    st.session_state.portal_calls = None

# ── Step 1: upload HTML ───────────────────────────────────────────────────────

st.subheader("Step 1 — Upload HTML files")
html_uploads = st.file_uploader(
    "Select one or more HTML exports from the EU portal",
    type=["html", "htm"],
    accept_multiple_files=True,
)

if st.button("Extract Calls", disabled=(not html_uploads)):
    all_calls: list[dict] = []
    log_lines: list[str] = []
    log_area = st.empty()

    for f in html_uploads:
        calls = extract_calls_from_html_bytes(f.read())
        log_lines.append(f"{f.name}: {len(calls)} calls found")
        log_area.code("\n".join(log_lines), language=None)
        all_calls.extend(calls)

    # Deduplicate
    seen, unique = set(), []
    for c in all_calls:
        k = (c["Call"], c["Title"])
        if k not in seen:
            seen.add(k)
            unique.append(c)

    if not unique:
        st.error(
            "No calls extracted. Make sure to save the portal page as "
            "'Complete Webpage' so that the call cards are present in the HTML."
        )
    else:
        st.session_state.portal_calls = unique
        st.success(f"Extracted {len(unique)} unique calls.")

# ── Step 2: mark interesting calls ───────────────────────────────────────────

if st.session_state.portal_calls:
    st.subheader("Step 2 — Mark interesting calls")
    st.markdown("Check the calls you want to mark as **Interesting**. Descriptions will be fetched for ALL calls.")

    calls = st.session_state.portal_calls
    selection_df = pd.DataFrame({
        "Interesting": [False] * len(calls),
        "Call ID": [c["Call"] for c in calls],
        "Title":   [c["Title"] for c in calls],
        "Deadline":[c["Deadline"] for c in calls],
    })

    edited = st.data_editor(
        selection_df,
        column_config={"Interesting": st.column_config.CheckboxColumn("Interesting ★")},
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
    )

    n_marked = int(edited["Interesting"].sum())
    st.caption(f"{n_marked} calls marked as interesting")

    # ── Step 3: fetch descriptions ────────────────────────────────────────────

    st.subheader("Step 3 — Fetch descriptions & export")
    if st.button("▶ Fetch Descriptions + Export Excel"):
        for i, call in enumerate(calls):
            call["Interesting brubotics"] = "Yes" if edited.iloc[i]["Interesting"] else ""

        df = pd.DataFrame(calls)[["Call", "Title", "URL", "Deadline",
                                   "Interesting brubotics", "Description", "Budget", "Type"]]

        progress_ph = st.progress(0)
        log_lines2: list[str] = ["Fetching SEDIA API data…"]
        log_ph = st.empty()
        log_ph.code("\n".join(log_lines2), language=None)

        df = asyncio.run(_fill_all_async(df, log_lines2, progress_ph, log_ph))

        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        buf.seek(0)

        st.success("Done!")
        st.download_button(
            "⬇ Download Excel",
            data=buf,
            file_name="Horizon_Calls.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.session_state.portal_calls = None
