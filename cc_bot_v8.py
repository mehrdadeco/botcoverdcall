# -*- coding: utf-8 -*-
"""
cc_bot_v8.py — اسکنِ کاوردکال (TSETMC) با محاسبهٔ «سود ماهانه» مطابق تعریف شما
- سود ماهانه: اگر «امروز» روز اعمال بود، سود/زیان پوشش چقدر بود نسبت به سرمایه (ساده)
  r_now = (پریمیوم − max(0, قیمت پایه − قیمت اعمال)) / (قیمت پایه − پریمیوم)
- سود سالانه: مرکب از سود ماهانهٔ بالا با دورهٔ DTE:
  r_annual = (1 + r_now) ** (365 / DTE) − 1
- دلتا: دلتا تقریبیِ موقعیت کاوردکال = 1 − دلتأ کال (Black‑Scholes با σ=0.35 و r=0)
- خروجی فقط فارسی

پیش‌نیاز:
  pip install python-telegram-bot==21.6 requests certifi pandas numpy

اجرا (PowerShell):
  $env:BOT_TOKEN=""
  python cc_bot_v8.py
"""
import os, time, logging, math
import requests, certifi
import numpy as np
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("cc-bot-v8")

URL = ""

# ===== فیلترهای ثابت (به‌صورت پیش‌فرض) =====
MIN_VOLUME = int(os.getenv("MIN_VOLUME", "50"))       # حداقل حجم
DTE_MIN = int(os.getenv("DTE_MIN", "31"))             # حداقل روز تا سررسید
MIN_BUFFER_PCT = float(os.getenv("MIN_BUFFER_PCT", "10.0"))  # حداقل حاشیه تا سر به سر (٪)
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "50.0"))  # حداکثر اسپرد٪
SIGMA = float(os.getenv("SIGMA", "0.35"))             # نوسان مفروض برای دلتا
R_RATE = 0.0                                          # نرخ بدون ریسک مفروض برای دلتا

def build_session():
    s = requests.Session()
    s.trust_env = False
    s.verify = certifi.where()
    s.headers.update({
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept":"application/json,text/plain,*/*",
        "Connection":"close",
    })
    retry = Retry(total=3, backoff_factor=0.6, status_forcelist=[429,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=4))
    return s

SESSION = build_session()

def fetch_raw_json_once():
    r = SESSION.get(URL, timeout=(5,20), proxies={"http": None, "https": None})
    r.raise_for_status()
    js = r.json()
    key = None
    for k in ["instrumentOptMarketWatch", "optMarketWatch", "data"]:
        if isinstance(js, dict) and k in js:
            key = k; break
    return js[key] if key else js

def extract_month_from_symbol(sym: str):
    try:
        s = str(sym)
        digits = ''.join(filter(str.isdigit, s[-5:]))
        if len(digits) >= 4:
            m = int(digits[:2])
            return m if 1 <= m <= 12 else np.nan
        return np.nan
    except:
        return np.nan

def unify_horizontal(payload) -> pd.DataFrame:
    raw = pd.json_normalize(payload, max_level=1)
    base_name_col = "lval30_UA" if "lval30_UA" in raw.columns else ("lVal30_UA" if "lVal30_UA" in raw.columns else None)
    rows = []
    for _, r in raw.iterrows():
        base_name = r.get(base_name_col) if base_name_col else ""
        ua_last   = r.get("pDrCotVal_UA"); ua_close  = r.get("pClosing_UA")
        ua_price  = ua_last if pd.notna(ua_last) else ua_close
        strike    = r.get("strikePrice")
        dte       = r.get("remainedDay")
        lot       = r.get("contractSize", 1000)

        # Call
        tkr_c = r.get("lVal18AFC_C")
        if pd.notna(tkr_c):
            rows.append({
                "نماد": tkr_c, "نوع": "کال",
                "نام سهم پایه": base_name,
                "قیمت سهم پایه": ua_price, "قیمت اعمال": strike,
                "آخرین معامله": r.get("pDrCotVal_C"), "قیمت پایانی": r.get("pClosing_C"),
                "بهترین خرید": r.get("pMeDem_C"), "بهترین فروش": r.get("pMeOf_C"),
                "حجم": r.get("qTotTran5J_C"),
                "روز تا سررسید": dte, "اندازه قرارداد": lot
            })
        # Put (نادیده می‌گیریم چون کاوردکال)
    df = pd.DataFrame(rows)
    for c in ["قیمت سهم پایه","قیمت اعمال","آخرین معامله","قیمت پایانی","بهترین خرید","بهترین فروش","حجم","اندازه قرارداد","روز تا سررسید"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if {"بهترین خرید","بهترین فروش"}.issubset(df.columns):
        df["میانگین مظنه"] = (df["بهترین خرید"].fillna(0) + df["بهترین فروش"].fillna(0)) / 2
        df["اسپرد"] = df["بهترین فروش"] - df["بهترین خرید"]
        df["اسپرد٪"] = np.where((df["بهترین خرید"]>0) & (df["بهترین فروش"]>0), df["اسپرد"] / np.maximum(df["میانگین مظنه"],1e-9) * 100, np.nan)

    if "نماد" in df.columns:
        df["برج انقضا"] = df["نماد"].apply(extract_month_from_symbol)

    return df

def pick_premium(row):
    p = row.get("آخرین معامله")
    if p is None or (isinstance(p, float) and np.isnan(p)) or (p<=0): p = row.get("قیمت پایانی")
    if p is None or (isinstance(p, float) and np.isnan(p)) or (p<=0):
        if "میانگین مظنه" in row and pd.notna(row["میانگین مظنه"]) and row["میانگین مظنه"]>0:
            p = row["میانگین مظنه"]
        else:
            bid = row.get("بهترین خرید"); ask = row.get("بهترین فروش")
            if pd.notna(bid) and pd.notna(ask) and bid>0 and ask>0:
                p = (bid+ask)/2
    try:
        return float(p or 0)
    except Exception:
        return 0.0

# دلتا تقریبیِ کال با Black-Scholes (r=0, σ ثابت)
def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def call_delta_bs(S, K, T, sigma, r=0.0):
    try:
        if S<=0 or K<=0 or T<=0 or sigma<=0: return float('nan')
        d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
        return _phi(d1)
    except Exception:
        return float('nan')

def scan_covered_calls(df_in: pd.DataFrame):
    if df_in is None or df_in.empty: return []
    calls = df_in[df_in["نوع"].astype(str)=="کال"].copy()
    if calls.empty: return []

    # فیلترهای ثابت
    if "روز تا سررسید" in calls.columns:
        calls = calls[calls["روز تا سررسید"].fillna(0).astype(int) >= DTE_MIN]
    if "حجم" in calls.columns:
        calls = calls[calls["حجم"].fillna(0).astype(float) >= MIN_VOLUME]
    if "اسپرد٪" in calls.columns:
        calls = calls[(calls["اسپرد٪"].abs() <= MAX_SPREAD_PCT) | calls["اسپرد٪"].isna()]

    نتایج = []
    for _, row in calls.iterrows():
        try:
            S0 = float(row.get("قیمت سهم پایه") or 0)
            K  = float(row.get("قیمت اعمال") or 0)
            dte = int((row.get("روز تا سررسید") or 0))
            prem = pick_premium(row)
            vol = float(row.get("حجم") or 0)
            if S0<=0 or K<=0 or prem<=0 or dte<=0:
                continue

            سرمایه = max(S0 - prem, 1e-9)

            # سود اگر امروز روز اعمال بود (پوششِ سهام + فروش اختیار)
            # سود = پریمیوم − max(0, S0 − K)
            سود_آنی = prem - max(0.0, S0 - K)
            بازده_آنی = سود_آنی / سرمایه   # «سود ماهانه» بر اساس تعریف شما

            # شرط حاشیه تا سر به سر
            حاشیه_BE = (prem / S0) * 100.0 if S0>0 else float('nan')
            if not np.isnan(حاشیه_BE) and حاشیه_BE < MIN_BUFFER_PCT:
                continue

            # سود سالانه مرکب از بازده همین دوره
            سالانه = (1.0 + بازده_آنی) ** (365.0 / dte) - 1.0

            # دلتا (تقریبی)، دلتأ موقعیت کاوردکال = 1 − دلتا کال
            T = dte / 365.0
            دلتا_کال = call_delta_bs(S0, K, T, SIGMA, R_RATE)
            دلتا_کاوردکال = (1.0 - دلتا_کال) if not np.isnan(دلتا_کال) else float('nan')

            نتایج.append({
                "پایه": row.get("نام سهم پایه"),
                "نماد": row.get("نماد"),
                "برج": row.get("برج انقضا"),
                "روز": dte,
                "قیمت پایه": S0,
                "اعمال": K,
                "پریمیوم": prem,
                "حجم": vol,
                "سود ماهانه": بازده_آنی,
                "سود سالانه": سالانه,
                "حاشیه BE": حاشیه_BE,
                "دلتا": دلتا_کاوردکال,
            })
        except Exception:
            continue

    # مرتب‌سازی بر اساس سود سالانه (بالاترین)
    نتایج.sort(key=lambda x: (x["سود سالانه"] if pd.notna(x["سود سالانه"]) else -1), reverse=True)
    return نتایج

def _fmt_num(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return "—"

def _fmt_pct(x, digits=1):
    try:
        return f"{x*100:.{digits}f}%"
    except Exception:
        return "—"

def _fmt_mon(m):
    try:
        return f"{int(m):02d}" if pd.notna(m) else "—"
    except Exception:
        return "—"

def build_message(items):
    if not items:
        return "چیزی مطابق فیلترها پیدا نشد."
    سرخط = f"📣 بهترین کاوردکال‌ها (حجم≥{MIN_VOLUME} | روز≥{DTE_MIN} | حاشیه BE≥{int(MIN_BUFFER_PCT)}٪ | اسپرد≤{int(MAX_SPREAD_PCT)}٪)"
    خطوط = [سرخط, ""]
    for i, it in enumerate(items[:10], 1):
        خطوط.append(f"{i}) خرید {it['پایه']} • فروش {it['نماد']} | برج {_fmt_mon(it['برج'])} | روز: {it['روز']}")
        خطوط.append(f"▫️ سود ماهانه: {_fmt_pct(it['سود ماهانه'],1)} | سود سالانه (مرکب): {_fmt_pct(it['سود سالانه'],0)} | دلتا: {it['دلتا']:.2f}")
        خطوط.append(f"▫️ پایه: {_fmt_num(it['قیمت پایه'])} | اعمال: {_fmt_num(it['اعمال'])} | پریمیوم: {_fmt_num(it['پریمیوم'])} | BE: {_fmt_num(it['قیمت پایه']-it['پریمیوم'])} | حاشیه BE: {it['حاشیه BE']:.1f}% | حجم: {_fmt_num(it['حجم'])}")
        خطوط.append("")
    return "\n".join(خطوط).strip()

# ===== Telegram =====
def main_menu():
    kb = [[InlineKeyboardButton("🔎 اسکن", callback_data="scan")]]
    return InlineKeyboardMarkup(kb)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await (update.message or update.callback_query.message).reply_text("سلام! روی «🔎 اسکن» بزن.", reply_markup=main_menu())

async def on_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer("در حال اسکن...")
    try:
        t0 = time.time()
        payload = fetch_raw_json_once()
        df = unify_horizontal(payload)
        items = scan_covered_calls(df)
        msg = build_message(items)
        msg += f"\n\n⏱ {time.time()-t0:.1f}s"
        await q.edit_message_text(msg, reply_markup=main_menu())
    except Exception as e:
        await q.edit_message_text(f"❌ خطا: {e}", reply_markup=main_menu())

def build_app():
    token = os.getenv("BOT_TOKEN", "8149405036:AAEHyxQzXTOjXTOpetAHPCYyyFYhPUfPPtM").strip()
    if not token:
        try:
            token = input("BOT_TOKEN:8149405036:AAEHyxQzXTOjXTOpetAHPCYyyFYhPUfPPtM ").strip()
        except Exception:
            token = ""
    if not token:
        raise SystemExit("❌ توکن لازم است.")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("id", lambda u,c: u.message.reply_text(str(u.effective_chat.id))))
    app.add_handler(CallbackQueryHandler(lambda u,c: on_scan(u,c) if u.callback_query.data=="scan" else None))
    return app

if __name__ == "__main__":
    app = build_app()
    log.info("Bot v8 running...")
    app.run_polling()
