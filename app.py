import hashlib
import hmac
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from modules.clustering import cluster_feedback, generate_theme_names
from modules.embeddings import generate_embeddings
from modules.preprocessing import preprocess_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "data" / "users.db"
PASSWORD_ITERATIONS = 300_000

st.set_page_config(
    page_title="Feedback-to-Feature Studio",
     page_icon="✦",
     layout="wide",
     initial_sidebar_state="collapsed",
)

st.markdown(
     """
     <style>
     :root {
          --ink: #f1f5ff;
          --muted: #91a0bd;
          --line: rgba(155, 177, 220, .15);
          --blue: #6aa7ff;
          --teal: #41d4c3;
          --violet: #9e7bff;
     }

     .stApp {
          background: #080d18;
          color: var(--ink);
          font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
     }
     .stApp::before {
          content: '';
          position: fixed;
          inset: 0;
          pointer-events: none;
          opacity: .35;
          background-image: linear-gradient(rgba(125, 154, 211, .045) 1px, transparent 1px),
                                linear-gradient(90deg, rgba(125, 154, 211, .045) 1px, transparent 1px);
          background-size: 54px 54px;
          mask-image: linear-gradient(to bottom, black, transparent 82%);
     }
     .stApp::after {
          content: '';
          position: fixed;
          width: 42vw;
          height: 42vw;
          top: -25vw;
          left: 38vw;
          pointer-events: none;
          background: radial-gradient(circle, rgba(105, 75, 205, .17), transparent 68%);
     }
     [data-testid='stHeader'], [data-testid='stToolbar'], footer { display: none; }
     .block-container { max-width: 1240px; padding: 4.5rem 3.5rem 3rem; }
     .auth-shell { animation: rise .7s ease both; }
     @keyframes rise { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
     .eyebrow { color: var(--teal); font-size: .7rem; letter-spacing: .2em; font-weight: 700; margin-bottom: 1.6rem; }
     .brand { display: flex; align-items: center; gap: 12px; color: var(--ink); font: 700 1rem 'Trebuchet MS', 'Segoe UI', sans-serif; }
     .brand-mark { position: relative; width: 28px; height: 28px; border: 1px solid rgba(106, 167, 255, .8); border-radius: 9px; transform: rotate(15deg); box-shadow: 0 0 22px rgba(65, 212, 195, .23); }
     .brand-mark::before, .brand-mark::after { content: ''; position: absolute; border-radius: 50%; background: var(--teal); box-shadow: 0 0 12px var(--teal); }
     .brand-mark::before { width: 6px; height: 6px; top: 5px; left: 5px; }
     .brand-mark::after { width: 5px; height: 5px; right: 4px; bottom: 5px; background: var(--violet); box-shadow: 0 0 12px var(--violet); }
     .hero-title { margin: 3.8rem 0 1.2rem; max-width: 600px; color: var(--ink); font: 800 clamp(2.8rem, 5vw, 5rem)/1.03 'Trebuchet MS', 'Segoe UI', sans-serif; letter-spacing: -.04em; }
     .hero-title span { color: var(--blue); }
     .hero-copy { max-width: 500px; color: var(--muted); font-size: 1.08rem; line-height: 1.7; }
     .signal-stage { position: relative; height: 290px; margin-top: 3rem; overflow: hidden; border: 1px solid rgba(138, 170, 226, .14); border-radius: 24px; background: linear-gradient(135deg, rgba(29, 47, 78, .48), rgba(17, 22, 43, .15)); box-shadow: inset 0 1px rgba(255,255,255,.06), 0 24px 80px rgba(0,0,0,.2); }
     .signal-stage::before { content: ''; position: absolute; inset: 0; background: radial-gradient(circle at 50% 48%, rgba(65, 212, 195, .12), transparent 31%); }
     .wire { position: absolute; height: 1px; transform-origin: left center; background: linear-gradient(90deg, rgba(106,167,255,.05), rgba(65,212,195,.55), rgba(158,123,255,.05)); box-shadow: 0 0 10px rgba(65,212,195,.35); }
     .wire.one { width: 220px; top: 92px; left: 28%; transform: rotate(28deg); }
     .wire.two { width: 260px; top: 164px; left: 39%; transform: rotate(-23deg); }
     .wire.three { width: 160px; top: 164px; left: 21%; transform: rotate(-5deg); }
     .node { position: absolute; width: 7px; height: 7px; border-radius: 50%; background: var(--teal); box-shadow: 0 0 0 5px rgba(65,212,195,.1), 0 0 18px var(--teal); }
     .node.a { top: 89px; left: 28%; } .node.b { top: 160px; left: 39%; background: var(--violet); box-shadow: 0 0 0 5px rgba(158,123,255,.1), 0 0 18px var(--violet); } .node.c { top: 160px; left: 21%; }
     .feedback { position: absolute; padding: 12px 15px; border: 1px solid rgba(177, 202, 246, .19); border-radius: 12px; color: #d9e3f9; background: rgba(44, 62, 96, .43); box-shadow: 0 12px 24px rgba(0,0,0,.18), inset 0 1px rgba(255,255,255,.06); backdrop-filter: blur(12px); font-size: .76rem; }
     .feedback.one { top: 28px; left: 7%; } .feedback.two { top: 54px; right: 7%; } .feedback.three { bottom: 28px; left: 7%; } .feedback.four { right: 8%; bottom: 31px; }
     .auth-card { margin: 2.5rem 0 0 auto; max-width: 430px; padding: 2.4rem; border: 1px solid var(--line); border-radius: 26px; background: linear-gradient(145deg, rgba(26, 38, 65, .78), rgba(14, 20, 37, .72)); box-shadow: 0 30px 90px rgba(0,0,0,.36), inset 0 1px rgba(255,255,255,.07); backdrop-filter: blur(22px); }
     .auth-card h2 { margin: 0 0 .55rem; color: var(--ink); font: 700 1.7rem 'Trebuchet MS', 'Segoe UI', sans-serif; }
     .auth-card p { color: var(--muted); line-height: 1.55; margin-bottom: 1.8rem; }
     label { color: #b8c5de !important; font-size: .82rem !important; }
     .stTextInput input { border: 1px solid rgba(155,177,220,.17); border-radius: 11px; color: var(--ink); background: rgba(7, 13, 27, .52); padding: .8rem 1rem; }
     .stTextInput input:focus { border-color: rgba(106,167,255,.8); box-shadow: 0 0 0 3px rgba(106,167,255,.1); }
     .stButton > button { width: 100%; border: 0; border-radius: 11px; padding: .78rem 1rem; color: #08121f; background: linear-gradient(100deg, var(--teal), #73bfff); font-weight: 700; box-shadow: 0 10px 25px rgba(65,212,195,.16); transition: transform .2s, box-shadow .2s; }
     .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 14px 30px rgba(65,212,195,.27); }
     .switch-copy { margin-top: 1.6rem; text-align: center; color: var(--muted); font-size: .86rem; }
     .switch-copy .stButton > button { width: auto; margin: -.3rem 0 0 .25rem; padding: 0; color: var(--blue); background: transparent; box-shadow: none; font-size: .86rem; }
     .stAlert { border-radius: 11px; }
     .workspace-shell { max-width: 1280px; margin: 0 auto; animation: rise .55s ease both; }
     .workspace-header { display: flex; align-items: center; justify-content: space-between; padding-bottom: 1.6rem; border-bottom: 1px solid var(--line); }
     .workspace-brand small { display: block; margin: 4px 0 0 40px; color: var(--muted); font-size: .58rem; letter-spacing: .18em; }
     .header-actions { display: flex; align-items: center; gap: .8rem; }
     .notification { display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 36px; border: 1px solid var(--line); border-radius: 50%; color: #b5c9ed; background: rgba(36, 51, 82, .42); font-size: 1rem; }
     .profile { display: flex; align-items: center; gap: .65rem; color: #dce7fb; font-size: .82rem; }
     .avatar { display: grid; place-items: center; width: 34px; height: 34px; border-radius: 50%; color: #07121f; background: linear-gradient(140deg, var(--teal), var(--blue)); font-size: .72rem; font-weight: 800; }
     .workspace-layout { display: grid; grid-template-columns: 205px minmax(0, 1fr); gap: 2.5rem; padding-top: 2rem; }
     .nav-panel { padding-right: 1.5rem; border-right: 1px solid var(--line); }
     .nav-label { margin: 0 0 1rem .65rem; color: #5d6c89; font-size: .62rem; letter-spacing: .18em; font-weight: 700; }
     .nav-item { display: flex; align-items: center; gap: .75rem; margin: .25rem 0; padding: .7rem .7rem; border-radius: 10px; color: #8e9db9; font-size: .84rem; }
     .nav-item.active { color: #e5f5ff; background: linear-gradient(90deg, rgba(65,212,195,.14), rgba(106,167,255,.08)); box-shadow: inset 3px 0 var(--teal); }
     .nav-icon { width: 19px; color: var(--blue); text-align: center; font-size: .9rem; }
     .workspace-main h1 { margin: 0; color: var(--ink); font: 800 clamp(1.8rem, 3vw, 2.8rem)/1.12 'Trebuchet MS', 'Segoe UI', sans-serif; letter-spacing: -.035em; }
     .workspace-main h1 span { color: var(--blue); }
     .dashboard-intro { display: flex; align-items: end; justify-content: space-between; gap: 1rem; margin-bottom: 2rem; }
     .dashboard-intro p { margin: .65rem 0 0; color: var(--muted); }
     .ready-pill { white-space: nowrap; padding: .45rem .75rem; border: 1px solid rgba(65,212,195,.25); border-radius: 999px; color: var(--teal); background: rgba(65,212,195,.07); font-size: .72rem; }
     .ready-dot { display: inline-block; width: 6px; height: 6px; margin-right: .4rem; border-radius: 50%; background: var(--teal); box-shadow: 0 0 9px var(--teal); }
     .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .9rem; margin-bottom: 2.3rem; }
     .metric-card { position: relative; min-height: 116px; padding: 1.15rem 1.2rem; overflow: hidden; border: 1px solid var(--line); border-radius: 16px; background: rgba(24, 36, 60, .62); box-shadow: inset 0 1px rgba(255,255,255,.04); }
     .metric-card::after { content: ''; position: absolute; right: -22px; bottom: -30px; width: 90px; height: 90px; border-radius: 50%; background: var(--metric-glow, rgba(106,167,255,.15)); filter: blur(18px); }
     .metric-label { color: #8998b6; font-size: .64rem; letter-spacing: .14em; font-weight: 700; }
     .metric-value { margin-top: .65rem; color: #f0f5ff; font: 800 2rem 'Trebuchet MS', 'Segoe UI', sans-serif; }
     .section-heading { margin: 2rem 0 1rem; }
     .section-heading h2 { margin: 0; color: #eaf1ff; font-size: 1.15rem; }
     .section-heading p { margin: .35rem 0 0; color: var(--muted); font-size: .8rem; }
     .surface { padding: 1.35rem; border: 1px solid var(--line); border-radius: 18px; background: rgba(20, 31, 53, .62); box-shadow: 0 18px 48px rgba(0,0,0,.14), inset 0 1px rgba(255,255,255,.035); }
     .feedback-list { display: grid; gap: .65rem; }
     .feedback-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: .85rem .95rem; border: 1px solid rgba(155,177,220,.1); border-radius: 12px; background: rgba(9, 16, 31, .38); }
     .feedback-quote { color: #dce6f8; font-size: .82rem; }
     .feedback-source { display: inline-block; margin-top: .35rem; color: #8292af; font-size: .68rem; }
     .source-badge { flex: 0 0 auto; padding: .3rem .55rem; border-radius: 999px; color: #9ed9ff; background: rgba(106,167,255,.1); font-size: .62rem; }
     .theme-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem; }
     .theme-card { padding: .95rem; border: 1px solid rgba(155,177,220,.11); border-radius: 13px; background: rgba(9, 16, 31, .34); }
     .theme-top { display: flex; justify-content: space-between; color: #dce6f8; font-size: .78rem; }
     .theme-count { color: var(--teal); font-size: .7rem; }
     .progress-track { height: 4px; margin-top: .8rem; overflow: hidden; border-radius: 4px; background: rgba(155,177,220,.13); }
     .progress-fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--teal), var(--blue)); }
     .opportunity-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; }
     .opportunity-card { padding: 1.15rem; border: 1px solid rgba(155,177,220,.13); border-radius: 15px; background: linear-gradient(145deg, rgba(37,52,83,.58), rgba(17,25,44,.5)); }
     .opportunity-card h3 { margin: 0 0 .85rem; color: #e8f0ff; font-size: .95rem; }
     .opportunity-card p { min-height: 48px; margin: 0; color: #94a4c0; font-size: .75rem; line-height: 1.55; }
     .priority { display: inline-block; margin-top: 1rem; padding: .3rem .55rem; border-radius: 999px; font-size: .62rem; }
     .priority.high { color: #ffb4be; background: rgba(239,101,122,.12); } .priority.medium { color: #c7b7ff; background: rgba(158,123,255,.12); }
     .quick-actions { display: grid; grid-template-columns: repeat(4, 1fr); gap: .7rem; }
     .workspace-main .stButton > button { border: 1px solid rgba(155,177,220,.16); color: #c7d6ef; background: rgba(30,45,74,.62); box-shadow: none; font-size: .76rem; }
     .workspace-main .stButton > button:hover { border-color: rgba(65,212,195,.55); color: #e8fffb; background: rgba(65,212,195,.1); }
     .nav-panel .stButton > button { margin: .17rem 0; padding: .65rem .7rem; border: 1px solid transparent; color: #8e9db9; background: transparent; box-shadow: none; text-align: left; font-size: .8rem; }
     .nav-panel .stButton > button:hover { color: #e5f5ff; background: rgba(106,167,255,.08); }
     .logout-button .stButton > button { width: auto; padding: .45rem .7rem; border: 1px solid var(--line); color: #9eabc3; background: transparent; box-shadow: none; font-size: .7rem; }
     .empty-state { padding: 3.5rem 2rem; text-align: center; border: 1px dashed rgba(155,177,220,.18); border-radius: 18px; background: rgba(20,31,53,.45); }
     .empty-icon { display: grid; place-items: center; width: 54px; height: 54px; margin: 0 auto 1rem; border: 1px solid rgba(65,212,195,.3); border-radius: 16px; color: var(--teal); background: rgba(65,212,195,.08); font-size: 1.35rem; }
     .empty-state h2 { margin: 0; color: #eaf1ff; font-size: 1.3rem; } .empty-state p { color: var(--muted); font-size: .84rem; }
     .workspace-footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line); color: #61718f; font-size: .7rem; }
     .sidebar-account { margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--line); }
     .sidebar-account-label { color: #5d6c89; font-size: .6rem; letter-spacing: .14em; }
     .sidebar-account-name { margin: .4rem 0 .8rem; overflow: hidden; color: #dce6f8; font-size: .78rem; text-overflow: ellipsis; white-space: nowrap; }
     .filter-row { display: grid; grid-template-columns: 1.5fr 1fr 1fr; gap: .7rem; margin-bottom: 1rem; }
     .filter-row label { display: none !important; }
     .filter-row .stTextInput input, .filter-row .stSelectbox div[data-baseweb='select'] { min-height: 38px; border-radius: 10px; background: rgba(9,16,31,.48); }
     .filter-row .stSelectbox > div { min-height: 38px; }
     .feedback-card-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: .75rem; }
     .feedback-card { min-height: 96px; padding: .85rem 1rem; border: 1px solid rgba(155,177,220,.12); border-radius: 14px; background: rgba(20,31,53,.62); }
     .feedback-card p { margin: 0 0 .8rem; color: #e1eafa; font-size: .86rem; line-height: 1.5; }
     .feedback-meta { display: flex; justify-content: space-between; gap: .5rem; color: #8292af; font-size: .68rem; }
     .feedback-pagination { padding: .65rem 0; color: #aebdd7; font-size: .76rem; text-align: center; }
     .theme-select-row { display: grid; grid-template-columns: minmax(0, 1fr) 1.4fr; gap: 1rem; align-items: start; }
     .theme-detail { min-height: 205px; padding: 1.25rem; border: 1px solid rgba(65,212,195,.2); border-radius: 15px; background: linear-gradient(145deg, rgba(30,58,79,.48), rgba(20,31,53,.62)); }
     .theme-detail h3 { margin: 0 0 .8rem; color: #eaf1ff; font-size: 1.1rem; }
     .detail-stat { margin-bottom: .75rem; color: #9baccc; font-size: .76rem; }
     .detail-stat strong { color: var(--teal); font-size: 1.1rem; }
     .detail-label { margin: 1rem 0 .35rem; color: #7d8eac; font-size: .64rem; letter-spacing: .12em; text-transform: uppercase; }
     .detail-copy { margin: 0; color: #d4def0; font-size: .78rem; line-height: 1.5; }
     .proposal-layout { display: grid; grid-template-columns: .72fr 1.28fr; gap: 1rem; align-items: start; }
     .proposal-card { padding: 1.35rem; border: 1px solid rgba(158,123,255,.22); border-radius: 16px; background: linear-gradient(145deg, rgba(42,37,79,.58), rgba(20,31,53,.62)); }
     .proposal-card h3 { margin: 0 0 1.25rem; color: #f0edff; font-size: 1.05rem; }
     .proposal-label { margin-top: 1rem; color: #9584d8; font-size: .63rem; letter-spacing: .13em; text-transform: uppercase; }
     .proposal-copy { margin: .35rem 0 0; color: #d9e1f3; font-size: .8rem; line-height: 1.55; }
     .criteria-list { margin: .35rem 0 0; padding-left: 1.1rem; color: #d9e1f3; font-size: .8rem; line-height: 1.7; }
     .document-card { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: 1.25rem; border: 1px solid rgba(65,212,195,.18); border-radius: 15px; background: rgba(20,31,53,.62); }
     .document-title { color: #eaf1ff; font-size: .9rem; font-weight: 700; }
     .document-copy { margin-top: .35rem; color: #8393b0; font-size: .73rem; line-height: 1.5; }
     .settings-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: .8rem; }
     .settings-card { padding: 1.15rem; border: 1px solid rgba(155,177,220,.12); border-radius: 14px; background: rgba(20,31,53,.62); }
     .settings-card h3 { margin: 0 0 .9rem; color: #e7efff; font-size: .9rem; }
     .setting-row { display: flex; justify-content: space-between; gap: 1rem; padding: .65rem 0; border-top: 1px solid rgba(155,177,220,.1); color: #91a0bd; font-size: .75rem; }
     .setting-row strong { color: #dce6f8; font-weight: 500; text-align: right; }
     .upload-note { margin-top: .8rem; color: #7384a1; font-size: .7rem; }
     .stCheckbox label { color: #c6d2e8 !important; }
     @media (max-width: 900px) { .workspace-layout { grid-template-columns: 1fr; gap: 1.5rem; } .nav-panel { display: flex; align-items: center; gap: .35rem; overflow-x: auto; padding: 0 0 .8rem; border-right: 0; border-bottom: 1px solid var(--line); } .nav-label { display: none; } .nav-panel .stButton > button { white-space: nowrap; } .metric-grid { grid-template-columns: repeat(2, 1fr); } }
     @media (max-width: 600px) { .block-container { padding: 2rem 1.2rem; } .hero-title { margin-top: 2.7rem; font-size: 3rem; } .signal-stage { height: 250px; } .feedback { font-size: .65rem; padding: 9px 10px; } .auth-card { margin-top: 2rem; padding: 1.6rem; } .workspace-header { align-items: flex-start; } .profile-name { display: none; } .dashboard-intro { display: block; } .ready-pill { display: inline-block; margin-top: 1rem; } .theme-grid, .opportunity-grid, .feedback-card-grid, .settings-grid { grid-template-columns: 1fr; } .quick-actions { grid-template-columns: repeat(2, 1fr); } .feedback-row { align-items: flex-start; flex-direction: column; gap: .55rem; } .filter-row, .theme-select-row, .proposal-layout { grid-template-columns: 1fr; } .document-card { align-items: flex-start; flex-direction: column; } }
     </style>
     """,
     unsafe_allow_html=True,
)

if "authenticated" not in st.session_state:
     st.session_state.authenticated = False
if "auth_mode" not in st.session_state:
     st.session_state.auth_mode = "login"
if "user_name" not in st.session_state:
     st.session_state.user_name = ""
if "user_email" not in st.session_state:
     st.session_state.user_email = ""
if "auth_message" not in st.session_state:
     st.session_state.auth_message = None


def init_database():
     """Create the local SQLite authentication database and users table."""
     DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
     with sqlite3.connect(DATABASE_PATH) as connection:
          connection.execute(
               """
               CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
               )
               """
          )


def hash_password(password):
     """Return a salted PBKDF2-HMAC-SHA256 password hash."""
     salt = secrets.token_bytes(16)
     digest = hashlib.pbkdf2_hmac(
          "sha256",
          password.encode("utf-8"),
          salt,
          PASSWORD_ITERATIONS,
     )
     return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password, stored_hash):
     """Safely compare a password with a stored salt and hash."""
     try:
          algorithm, iterations, salt_hex, digest_hex = stored_hash.split("$")
          if algorithm != "pbkdf2_sha256":
               return False
          candidate = hashlib.pbkdf2_hmac(
               "sha256",
               password.encode("utf-8"),
               bytes.fromhex(salt_hex),
               int(iterations),
          ).hex()
          return hmac.compare_digest(candidate, digest_hex)
     except (ValueError, TypeError):
          return False


def create_user(name, email, password):
     """Insert a new account, returning False when the email already exists."""
     try:
          with sqlite3.connect(DATABASE_PATH) as connection:
               connection.execute(
                    "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (
                         name.strip(),
                         email.strip().lower(),
                         hash_password(password),
                         datetime.now(timezone.utc).isoformat(),
                    ),
               )
          return True
     except sqlite3.IntegrityError:
          return False


def get_user(email):
     """Fetch one account by normalized email for login verification."""
     with sqlite3.connect(DATABASE_PATH) as connection:
          connection.row_factory = sqlite3.Row
          row = connection.execute(
               "SELECT name, email, password_hash FROM users WHERE email = ?",
               (email.strip().lower(),),
          ).fetchone()
     return dict(row) if row else None


def valid_email(email):
     return bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email.strip()))


def logout_user():
     st.session_state.authenticated = False
     st.session_state.user_name = ""
     st.session_state.user_email = ""
     st.session_state.auth_mode = "login"
     st.session_state.active_page = "Overview"
     st.session_state.auth_message = None


init_database()


def load_feedback_data():
     required_columns = {"id", "source", "feedback", "product_area"}
     try:
          feedback_df = pd.read_csv("data/feedback.csv")
     except FileNotFoundError:
          return pd.DataFrame(columns=sorted(required_columns))
     except (pd.errors.EmptyDataError, pd.errors.ParserError):
          return pd.DataFrame(columns=sorted(required_columns))

     missing_columns = required_columns.difference(feedback_df.columns)
     if missing_columns:
          return pd.DataFrame(columns=sorted(required_columns))

     return feedback_df


@st.cache_data
def load_knowledge_base():
     """Load the local product knowledge document without changing it."""
     knowledge_path = PROJECT_ROOT / "data" / "knowledge_base.txt"
     try:
          return knowledge_path.read_text(encoding="utf-8")
     except OSError:
          return "The product knowledge document is currently unavailable."


@st.cache_data(show_spinner="Analyzing customer feedback...")
def analyze_feedback(feedback_df):
     """Preprocess, embed, cluster, and label feedback in memory."""
     analyzed = preprocess_dataframe(feedback_df)
     if analyzed.empty:
          analyzed["cluster"] = pd.Series(dtype="int64")
          analyzed["theme"] = pd.Series(dtype="object")
          return analyzed

     embeddings = generate_embeddings(analyzed["cleaned_feedback"].tolist())
     labels, _ = cluster_feedback(embeddings)
     theme_names = generate_theme_names(analyzed, labels)
     analyzed["cluster"] = labels
     analyzed["theme"] = [theme_names[int(label)] for label in labels]
     return analyzed


def show_product_visual():
     st.markdown(
          """
          <div class="signal-stage">
               <div class="wire one"></div><div class="wire two"></div><div class="wire three"></div>
               <div class="node a"></div><div class="node b"></div><div class="node c"></div>
               <div class="feedback one">Dashboard is too slow</div>
               <div class="feedback two">Please add dark mode</div>
               <div class="feedback three">Need better notifications</div>
               <div class="feedback four">Reports should be easier to export</div>
          </div>
          """,
          unsafe_allow_html=True,
     )


def switch_mode(mode):
     st.session_state.auth_mode = mode


def show_auth():
     mode = st.session_state.auth_mode
     left, right = st.columns([1.18, .82], gap="large")
     with left:
          st.markdown(
               """
               <div class="auth-shell">
                    <div class="brand"><span class="brand-mark"></span> Feedback-to-Feature Studio</div>
                    <div class="eyebrow" style="margin-top: 4.5rem;">PRODUCT INTELLIGENCE WORKSPACE</div>
                    <h1 class="hero-title">Feedback-to-<br><span>Feature Studio</span></h1>
                    <div class="hero-copy">Turn customer voice into product action.<br>Transform scattered customer feedback into clear, actionable product opportunities.</div>
               </div>
               """,
               unsafe_allow_html=True,
          )
          show_product_visual()
     with right:
          st.markdown('<div class="auth-card">', unsafe_allow_html=True)
          if mode == "login":
               st.markdown("<h2>Welcome back</h2><p>Sign in to your workspace</p>", unsafe_allow_html=True)
               if st.session_state.auth_message:
                    st.success(st.session_state.auth_message)
                    st.session_state.auth_message = None
               email = st.text_input("Email address", key="login_email", placeholder="you@company.com")
               password = st.text_input("Password", key="login_password", type="password", placeholder="Enter your password")
               if st.button("Sign In", type="primary", key="sign_in"):
                    account = get_user(email)
                    if account and verify_password(password, account["password_hash"]):
                         st.session_state.authenticated = True
                         st.session_state.user_name = account["name"]
                         st.session_state.user_email = account["email"]
                         st.session_state.active_page = "Overview"
                         st.rerun()
                    else:
                         st.error("Incorrect email or password.")
               st.markdown('<div class="switch-copy">Forgot password?</div>', unsafe_allow_html=True)
               st.markdown('<div class="switch-copy">Don\'t have an account?</div>', unsafe_allow_html=True)
               if st.button("Create account", key="to_signup"):
                    switch_mode("signup")
                    st.rerun()
          else:
               st.markdown("<h2>Create account</h2><p>Start turning customer feedback into product opportunities.</p>", unsafe_allow_html=True)
               name = st.text_input("Full name", key="signup_name", placeholder="Your name")
               email = st.text_input("Work email", key="signup_email", placeholder="you@company.com")
               password = st.text_input("Password", key="signup_password", type="password", placeholder="Create a password")
               confirmation = st.text_input("Confirm password", key="signup_confirmation", type="password", placeholder="Repeat your password")
               if st.button("Create account", type="primary", key="create_account"):
                    if not name.strip() or not email.strip() or not password:
                         st.error("Complete all fields to continue.")
                    elif password != confirmation:
                         st.error("Passwords do not match.")
                    elif not valid_email(email):
                         st.error("Enter a valid email address.")
                    elif not create_user(name, email, password):
                         st.error("An account with this email already exists. Please sign in.")
                    else:
                         st.session_state.auth_mode = "login"
                         st.session_state.auth_message = "Account created successfully. You can now sign in."
                         st.rerun()
               st.markdown('<div class="switch-copy">Already have an account?</div>', unsafe_allow_html=True)
               if st.button("Sign in", key="to_login"):
                    switch_mode("login")
                    st.rerun()
          st.markdown('</div>', unsafe_allow_html=True)


def render_header(account):
     name = account.get("name", "Workspace owner") if account else "Workspace owner"
     initials = "".join(part[0] for part in name.split()[:2]).upper() or "WO"
     st.markdown(
          f"""
          <div class="workspace-header">
               <div class="workspace-brand">
                    <div class="brand"><span class="brand-mark"></span> Feedback-to-Feature Studio</div>
                    <small>PRODUCT INTELLIGENCE</small>
               </div>
               <div class="header-actions">
                    <span class="notification" title="Notifications">◌</span>
                    <div class="profile"><span class="avatar">{initials}</span><span class="profile-name">{name}</span></div>
                    <div class="logout-button">
          """,
          unsafe_allow_html=True,
     )
     if st.button("Logout", key="logout"):
          logout_user()
          st.rerun()
     st.markdown("</div></div></div>", unsafe_allow_html=True)


def render_sidebar(account):
     pages = [("Overview", "⌂"), ("Feedback", "▤"), ("Insights", "◈"), ("Feature Lab", "✦"), ("Knowledge Base", "▧"), ("Settings", "⚙")]
     active_page = st.session_state.get("active_page", "Overview")
     st.markdown('<div class="nav-label">WORKSPACE</div>', unsafe_allow_html=True)
     for page, icon in pages:
          if page == active_page:
               st.markdown(f'<div class="nav-item active"><span class="nav-icon">{icon}</span>{page}</div>', unsafe_allow_html=True)
          else:
               if st.button(f"{icon}   {page}", key=f"nav_{page}"):
                    st.session_state.active_page = page
                    st.rerun()
     name = account.get("name", "Workspace owner") if account else "Workspace owner"
     st.markdown(f'<div class="sidebar-account"><div class="sidebar-account-label">LOGGED IN AS</div><div class="sidebar-account-name">{name}</div></div>', unsafe_allow_html=True)
     if st.button("Logout", key="sidebar_logout"):
          logout_user()
          st.rerun()


def render_metric_card(label, value, glow):
     st.markdown(
          f'<div class="metric-card" style="--metric-glow: {glow};"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
          unsafe_allow_html=True,
     )


def render_feedback_card(quote, source):
     st.markdown(
          f'<div class="feedback-row"><div><div class="feedback-quote">"{quote}"</div><span class="feedback-source">Source: {source}</span></div><span class="source-badge">{source}</span></div>',
          unsafe_allow_html=True,
     )


def render_theme_card(name, mentions, width, percentage=None):
     count_label = f"{mentions} mentions"
     if percentage is not None:
          count_label += f" · {percentage:.0f}%"
     st.markdown(
          f'<div class="theme-card"><div class="theme-top"><span>{name}</span><span class="theme-count">{count_label}</span></div><div class="progress-track"><div class="progress-fill" style="width:{width}%"></div></div></div>',
          unsafe_allow_html=True,
     )


def render_feature_card(title, need, priority):
     priority_class = priority.lower()
     st.markdown(
          f'<div class="opportunity-card"><h3>{title}</h3><p><strong>Customer need:</strong><br>{need}</p><span class="priority {priority_class}">{priority} priority</span></div>',
          unsafe_allow_html=True,
     )


def render_section_heading(title, subtitle):
     st.markdown(f'<div class="section-heading"><h2>{title}</h2><p>{subtitle}</p></div>', unsafe_allow_html=True)


def show_dashboard(account):
     name = account.get("name", "there") if account else "there"
     st.markdown(f'<div class="dashboard-intro"><div><h1>Good morning, <span>{name}.</span></h1><p>Turn customer feedback into clear product opportunities.</p></div><div class="ready-pill"><span class="ready-dot"></span>Workspace ready</div></div>', unsafe_allow_html=True)
     st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
     metrics = [("TOTAL FEEDBACK", "38", "rgba(106,167,255,.2)"), ("RECURRING THEMES", "7", "rgba(65,212,195,.18)"), ("FEATURE IDEAS", "12", "rgba(158,123,255,.2)"), ("HIGH PRIORITY", "4", "rgba(239,101,122,.18)")]
     metric_columns = st.columns(4, gap="small")
     for column, (label, value, glow) in zip(metric_columns, metrics):
          with column:
               render_metric_card(label, value, glow)
     st.markdown('</div>', unsafe_allow_html=True)

     render_section_heading("Recent Activity", "A quick pulse from your workspace")
     st.markdown('<div class="surface"><div class="feedback-list">', unsafe_allow_html=True)
     for quote, source in [("New feedback collection is ready to review.", "Today · Inbox"), ("Performance emerged as a recurring theme.", "Yesterday · Insights"), ("Faster Dashboard Experience was added.", "Yesterday · Feature Lab")]:
          render_feedback_card(quote, source)
     st.markdown('</div></div>', unsafe_allow_html=True)

     render_section_heading("Top Themes", "The strongest signals from customer voice")
     st.markdown('<div class="theme-grid">', unsafe_allow_html=True)
     for name, mentions, width in [("Performance", 8, 88), ("User Experience", 6, 68), ("Notifications", 5, 56)]:
          render_theme_card(name, mentions, width)
     st.markdown('</div>', unsafe_allow_html=True)

     render_section_heading("Quick actions", "Start with a focused workspace task")
     quick_actions = ["Analyze Feedback", "Explore Insights", "Create Feature"]
     action_columns = st.columns(3, gap="small")
     for column, action in zip(action_columns, quick_actions):
          with column:
               if st.button(action, key=f"quick_{action}"):
                    target = {"Analyze Feedback": "Feedback", "Explore Insights": "Insights", "Create Feature": "Feature Lab"}[action]
                    st.session_state.active_page = target
                    st.rerun()


def render_page_header(title, subtitle):
     st.markdown(f'<div class="dashboard-intro"><div><h1>{title}</h1><p>{subtitle}</p></div><div class="ready-pill"><span class="ready-dot"></span>Workspace ready</div></div>', unsafe_allow_html=True)


def show_feedback_page():
     render_page_header("Feedback Inbox", "Review and explore customer feedback.")
     feedback_df = load_feedback_data()
     if feedback_df.empty:
          st.warning("No feedback data is currently available in the included workspace dataset.")

     source_options = ["All sources"] + sorted(feedback_df["source"].dropna().astype(str).unique().tolist())
     product_area_options = ["All product areas"] + sorted(feedback_df["product_area"].dropna().astype(str).unique().tolist())
     st.markdown('<div class="filter-row">', unsafe_allow_html=True)
     search = st.text_input("Search feedback", placeholder="Search feedback...", key="feedback_search")
     source = st.selectbox("Source", source_options, key="feedback_source")
     area = st.selectbox("Product area", product_area_options, key="feedback_product_area")
     st.markdown('</div>', unsafe_allow_html=True)
     filtered_df = feedback_df.copy()
     if search:
          filtered_df = filtered_df[filtered_df["feedback"].astype(str).str.contains(search, case=False, na=False, regex=False)]
     if source != "All sources":
          filtered_df = filtered_df[filtered_df["source"].astype(str) == source]
     if area != "All product areas":
          filtered_df = filtered_df[filtered_df["product_area"].astype(str) == area]

     filter_signature = (search, source, area)
     if st.session_state.get("feedback_filter_signature") != filter_signature:
          st.session_state.feedback_page = 1
          st.session_state.feedback_filter_signature = filter_signature

     page_size = 6
     total_items = len(filtered_df)
     total_pages = max(1, (total_items + page_size - 1) // page_size)
     current_page = min(st.session_state.get("feedback_page", 1), total_pages)
     st.session_state.feedback_page = current_page
     start_index = (current_page - 1) * page_size
     page_df = filtered_df.iloc[start_index:start_index + page_size]

     st.markdown(f'<div class="section-heading"><h2>{len(filtered_df)} feedback items</h2><p>Customer feedback from your workspace</p></div>', unsafe_allow_html=True)
     if filtered_df.empty:
          st.markdown('<div class="empty-state"><div class="empty-icon">⌕</div><h2>No matching feedback found.</h2><p>Try changing your search or filters.</p></div>', unsafe_allow_html=True)
     else:
          st.markdown('<div class="feedback-card-grid">', unsafe_allow_html=True)
          for _, record in page_df.iterrows():
               feedback_text = str(record["feedback"])
               item_source = str(record["source"])
               product_area = str(record["product_area"])
               st.markdown(f'<div class="feedback-card"><p>"{feedback_text}"</p><div class="feedback-meta"><span>{item_source}</span><span>{product_area}</span></div></div>', unsafe_allow_html=True)
          st.markdown('</div>', unsafe_allow_html=True)

          previous_column, status_column, next_column = st.columns([1, 1, 1], gap="small")
          with previous_column:
               if st.button("← Previous", key="feedback_previous", disabled=current_page == 1):
                    st.session_state.feedback_page -= 1
                    st.rerun()
          with status_column:
               st.markdown(f'<div class="feedback-pagination">Page {current_page} of {total_pages}</div>', unsafe_allow_html=True)
          with next_column:
               if st.button("Next →", key="feedback_next", disabled=current_page == total_pages):
                    st.session_state.feedback_page += 1
                    st.rerun()
     render_section_heading("Bring in more customer voice", "Your workspace currently uses the included feedback dataset.")
     if st.button("Upload Feedback", key="upload_feedback"):
          st.info("Upload capability will be enabled in the next stage.")


def show_insights_page():
     render_page_header("Customer Insights", "Discover recurring customer needs and pain points.")
     feedback_df = load_feedback_data()
     analyzed_df = analyze_feedback(feedback_df)
     total_feedback = len(analyzed_df)
     theme_counts = analyzed_df["theme"].value_counts().sort_values(ascending=False) if total_feedback else pd.Series(dtype="int64")
     theme_names = theme_counts.index.tolist()

     metric_columns = st.columns(2, gap="small")
     with metric_columns[0]:
          render_metric_card("TOTAL FEEDBACK", str(total_feedback), "rgba(106,167,255,.2)")
     with metric_columns[1]:
          render_metric_card("RECURRING THEMES", str(len(theme_counts)), "rgba(65,212,195,.18)")

     if not total_feedback:
          st.warning("No feedback data is currently available for insights.")
          return

     max_mentions = max(theme_counts)
     st.markdown('<div class="theme-grid">', unsafe_allow_html=True)
     for name, mentions in theme_counts.items():
          percentage = mentions / total_feedback * 100
          width = mentions / max_mentions * 100
          render_theme_card(name, int(mentions), width, percentage)
     st.markdown('</div>', unsafe_allow_html=True)
     render_section_heading("Theme details", "Select a theme to inspect the customer signal behind it.")
     selected = st.selectbox("Theme", theme_names, label_visibility="collapsed", key="insights_theme")
     selected_feedback = analyzed_df[analyzed_df["theme"] == selected]
     selected_count = len(selected_feedback)
     common_area = selected_feedback["product_area"].mode().iat[0] if not selected_feedback.empty else "Customer needs"
     concern = selected_feedback["feedback"].iloc[0] if not selected_feedback.empty else "No feedback is available for this theme."
     feedback_items = "".join(f"<li>{str(text)}</li>" for text in selected_feedback["feedback"].tolist())
     st.markdown(
          f'<div class="theme-detail"><h3>{selected}</h3><div class="detail-stat"><strong>{selected_count}</strong> related feedback items</div><div class="detail-label">Common customer concern</div><p class="detail-copy">{concern}</p><div class="detail-label">Product area</div><p class="detail-copy">{common_area}</p><div class="detail-label">Feedback in this theme</div><ul class="criteria-list">{feedback_items}</ul></div>',
          unsafe_allow_html=True,
     )


def show_feature_lab_page():
     render_page_header("Feature Lab", "Turn customer needs into structured product requirements.")
     selected = st.selectbox("Select an insight", ["Performance", "User Experience", "Notifications", "Reporting", "Mobile Experience", "Search"])
     if st.button("Generate Feature Proposal", key="generate_proposal", type="primary"):
          st.session_state.proposal_generated = True
     if st.session_state.get("proposal_generated"):
          render_section_heading("Feature Proposal", "A demo proposal based on the selected customer insight.")
          proposals = {
               "Performance": {"title": "Faster Dashboard Experience", "problem": "Users wait too long for dashboard information.", "need": "Customers need a fast, dependable view of their most important data.", "solution": "Improve dashboard loading and prioritize key information for the first view.", "priority": "High", "impact": "Reduce waiting during daily workflows and improve confidence in the dashboard."},
               "User Experience": {"title": "Comfortable Workspace Themes", "problem": "Users want a calmer interface that fits their working environment.", "need": "Customers need display preferences that make long work sessions more comfortable.", "solution": "Add dark mode and clear display preferences to the workspace.", "priority": "Medium", "impact": "Improve comfort, focus, and perceived product quality.",},
               "Notifications": {"title": "Smart Notifications", "problem": "Users miss important activity or receive too many low-value alerts.", "need": "Customers need timely updates with better control over notification frequency.", "solution": "Provide configurable notification types and digest preferences.", "priority": "Medium", "impact": "Increase the usefulness of alerts while reducing notification fatigue."},
               "Reporting": {"title": "Flexible Report Delivery", "problem": "Customers cannot easily export or schedule the reports they need to share.", "need": "Customers need reports in the format and cadence that fits their workflow.", "solution": "Add flexible export options and scheduled report delivery.", "priority": "High", "impact": "Save teams time and make reporting easier to share with stakeholders."},
               "Mobile Experience": {"title": "Reliable Mobile Reporting", "problem": "Mobile users experience crashes and missing report capabilities.", "need": "Customers need dependable access to important product information on mobile.", "solution": "Improve mobile stability and expand report support on iOS and Android.", "priority": "High", "impact": "Make mobile access more reliable for teams working away from their desks."},
               "Search": {"title": "More Reliable Search", "problem": "Search misses exact matches and struggles with special characters.", "need": "Customers need to find projects, reports, and users quickly.", "solution": "Improve matching behavior and make search filters more dependable.", "priority": "Medium", "impact": "Reduce time spent manually searching and increase discoverability."},
          }
          proposal = proposals[selected]
          st.markdown('<div class="proposal-layout">', unsafe_allow_html=True)
          st.markdown(f'<div class="proposal-card"><h3>{proposal["title"]}</h3><div class="priority {proposal["priority"].lower()}">{proposal["priority"]} priority</div><div class="proposal-label">Customer Problem</div><p class="proposal-copy">{proposal["problem"]}</p><div class="proposal-label">Customer Need</div><p class="proposal-copy">{proposal["need"]}</p><div class="proposal-label">Proposed Solution</div><p class="proposal-copy">{proposal["solution"]}</p></div>', unsafe_allow_html=True)
          st.markdown(f'<div class="proposal-card"><div class="proposal-label">Priority</div><p class="proposal-copy">{proposal["priority"]}</p><div class="proposal-label">Expected Impact</div><p class="proposal-copy">{proposal["impact"]}</p><div class="proposal-label">User Story</div><p class="proposal-copy">As a customer, I want a better {selected.lower()} experience so I can complete important work with less friction.</p><div class="proposal-label">Acceptance Criteria</div><ul class="criteria-list"><li>Core experience is clear and easy to discover.</li><li>Primary action completes with minimal friction.</li><li>Existing workspace behavior remains consistent.</li></ul></div>', unsafe_allow_html=True)
          st.markdown('</div>', unsafe_allow_html=True)


def show_knowledge_page():
     render_page_header("Product Knowledge", "Manage the product information used to create contextual feature proposals.")
     knowledge_text = load_knowledge_base()
     st.markdown('<div class="document-card"><div><div class="document-title">Product knowledge base</div><div class="document-copy">Loaded from data/knowledge_base.txt.<br>Product overview, capabilities, limitations, and supported platforms.</div></div><span class="source-badge">TXT</span></div>', unsafe_allow_html=True)
     with st.expander("View current product knowledge"):
          st.text(knowledge_text)
     st.markdown('<div class="settings-grid">', unsafe_allow_html=True)
     for title, copy in [("Product Overview", "Feedback-to-Feature Studio helps product teams turn customer voice into clear opportunities."), ("Current Capabilities", "Review feedback, explore themes, and shape structured feature ideas."), ("Known Limitations", "Some workspace capabilities are being prepared for the next stage."), ("Supported Platforms", "Designed for modern desktop workflows with responsive layouts for smaller screens.")]:
          st.markdown(f'<div class="settings-card"><h3>{title}</h3><p class="detail-copy">{copy}</p></div>', unsafe_allow_html=True)
     st.markdown('</div>', unsafe_allow_html=True)
     if st.button("Upload Knowledge", key="upload_knowledge"):
          st.info("Knowledge upload will be enabled in the next stage.")
     st.markdown('<div class="upload-note">Document processing is not enabled in this prototype.</div>', unsafe_allow_html=True)


def show_settings_page(account):
     render_page_header("Workspace Settings", "Configure your workspace preferences.")
     name = account.get("name", "Workspace owner") if account else "Workspace owner"
     email = account.get("email", "No email") if account else "No email"
     st.markdown('<div class="settings-grid">', unsafe_allow_html=True)
     st.markdown(f'<div class="settings-card"><h3>Profile</h3><div class="setting-row"><span>Name</span><strong>{name}</strong></div><div class="setting-row"><span>Email</span><strong>{email}</strong></div></div>', unsafe_allow_html=True)
     st.markdown('<div class="settings-card"><h3>Appearance</h3><div class="setting-row"><span>Theme</span><strong>Midnight workspace</strong></div><div class="setting-row"><span>Density</span><strong>Comfortable</strong></div></div>', unsafe_allow_html=True)
     st.markdown('<div class="settings-card"><h3>AI Configuration</h3><div class="setting-row"><span>Status</span><strong>Coming in the next stage</strong></div><div class="setting-row"><span>Workspace mode</span><strong>Demo experience</strong></div></div>', unsafe_allow_html=True)
     st.markdown('<div class="settings-card"><h3>Account</h3><div class="setting-row"><span>Access</span><strong>Personal workspace</strong></div><div class="setting-row"><span>Session</span><strong>Active</strong></div></div>', unsafe_allow_html=True)
     st.markdown('</div>', unsafe_allow_html=True)
     if st.button("Logout", key="settings_logout"):
          logout_user()
          st.rerun()


def show_workspace():
     if "active_page" not in st.session_state:
          st.session_state.active_page = "Overview"
     account = {
          "name": st.session_state.get("user_name", "Workspace owner"),
          "email": st.session_state.get("user_email", ""),
     }
     st.markdown('<div class="workspace-shell">', unsafe_allow_html=True)
     render_header(account)
     sidebar, main = st.columns([.19, .81], gap="large")
     with sidebar:
          st.markdown('<div class="nav-panel">', unsafe_allow_html=True)
          render_sidebar(account)
          st.markdown('</div>', unsafe_allow_html=True)
     with main:
          st.markdown('<div class="workspace-main">', unsafe_allow_html=True)
          page = st.session_state.active_page
          if page == "Overview":
               show_dashboard(account)
          elif page == "Feedback":
               show_feedback_page()
          elif page == "Insights":
               show_insights_page()
          elif page == "Feature Lab":
               show_feature_lab_page()
          elif page == "Knowledge Base":
               show_knowledge_page()
          elif page == "Settings":
               show_settings_page(account)
          st.markdown('<div class="workspace-footer">Feedback-to-Feature Studio · Product intelligence workspace</div>', unsafe_allow_html=True)
          st.markdown('</div>', unsafe_allow_html=True)
     st.markdown('</div>', unsafe_allow_html=True)


if st.session_state.authenticated:
     show_workspace()
else:
     show_auth()
