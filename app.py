# ==============================================================================
# CHURN / AI — Customer Retention Intelligence Agent  |  ONE Google Colab cell
# Predict risk. Understand behavior. Retain intelligently.
# ==============================================================================
import subprocess, sys, os, re, zipfile, html, warnings
warnings.filterwarnings("ignore")

def _deps():
    try:
        from importlib.metadata import version
        if int(version("gradio").split(".")[0]) not in (4, 5): raise RuntimeError("gradio version")
        for p in ("plotly", "scipy", "scikit-learn", "requests", "pandas", "numpy"): version(p)
    except Exception:
        print("Installing dependencies (about a minute)...")
        try: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio>=4.44,<6", "plotly", "scipy", "scikit-learn>=1.2", "requests", "pandas", "numpy"])
        except Exception as e: print("Install warning:", type(e).__name__, "- check internet access and re-run.")
_deps()
try:
    import numpy as np, pandas as pd, plotly.graph_objects as go, gradio as gr, requests
    from scipy import stats
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix,
                                 roc_curve, precision_recall_curve, brier_score_loss, silhouette_score)
    from sklearn.inspection import permutation_importance
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
except Exception as e:
    raise SystemExit("Libraries could not be imported (" + type(e).__name__ + "). Restart the runtime and re-run this cell.")

C = dict(bg="#101722", sf="#18212D", te="#42C9B4", mi="#7BE1C8", co="#FF8E86", am="#F4BC77", tx="#F3F7FA", mu="#9DADBC")
RISKS = ["Critical", "High", "Moderate", "Low"]
RC = {"Critical": C["co"], "High": C["am"], "Moderate": C["mi"], "Low": C["te"], "Unscored": C["mu"]}
BADGE = {"Critical": "🔴 Critical", "High": "🟠 High", "Moderate": "🟡 Moderate", "Low": "🟢 Low", "Unscored": "⚪ Unscored"}
OBJS = ["Balanced (max F1)", "Prioritise recall (max F2)", "Cost-based (missed churner vs false alarm)"]
G = {}
TESTS = []
BLANK = ["", "nan", "none", "n/a", "na", "-", "null"]
esc = lambda x: html.escape(str(x))
norm = lambda s: re.sub(r"\W+", "_", str(s).strip().lower()).strip("_")
nul = lambda v: v is None or (not isinstance(v, str) and bool(pd.isna(v)))
money = lambda v: "—" if nul(v) else ("-$%s" % format(abs(v), ",.0f") if v < 0 else "$" + format(v, ",.0f"))
pc0 = lambda v: "—" if nul(v) else format(v, ".0%")
pc1 = lambda v: "—" if nul(v) else format(v, ".1%")
fx = lambda v, d=2: "—" if nul(v) else format(v, "." + str(d) + "f")
n0f = lambda v: "—" if nul(v) else format(v, ",.0f")
NA = lambda n: tuple(gr.update() for _ in range(n))

SYN = {
 "customer_id": ["customerid", "cust_id", "user_id", "account_id", "id"],
 "tenure_months": ["tenure", "months_active", "customer_tenure"],
 "subscription_start_date": ["start_date", "signup_date", "join_date"],
 "contract_type": ["contract", "contract_term"], "subscription_plan": ["plan", "plan_type", "tier"],
 "monthly_fee": ["monthly_charges", "monthly_price", "mrr", "price", "monthly_revenue"],
 "customer_lifetime_value": ["clv", "ltv"], "historical_revenue": ["total_revenue", "total_charges", "lifetime_revenue"],
 "payment_failures": ["failed_payments", "payment_failed", "failed_charges"],
 "days_since_last_login": ["last_login_days", "recency_days", "days_inactive"],
 "logins_last_30_days": ["logins_30d", "logins", "sessions_30d", "login_count_30d"],
 "usage_minutes": ["minutes_used", "usage", "watch_time_minutes"],
 "usage_change_percentage": ["usage_change", "usage_change_pct", "usage_trend"],
 "features_used": ["feature_count", "features_adopted", "num_features"],
 "support_tickets": ["tickets", "support_contacts", "num_tickets"],
 "unresolved_complaints": ["open_complaints", "unresolved_tickets", "open_tickets"],
 "customer_satisfaction": ["csat", "satisfaction", "satisfaction_score"], "nps": ["nps_score"],
 "email_engagement": ["email_open_rate", "email_ctr"], "campaign_interactions": ["campaign_clicks", "marketing_interactions"],
 "discount_usage": ["discount_used", "promo_used", "discount_pct", "discount_count"],
 "renewal_date": ["next_renewal_date", "contract_end_date"], "days_to_renewal": ["days_until_renewal"],
 "cancellation_date": ["churn_date", "cancel_date", "end_date"],
 "churned": ["churn", "is_churned", "cancelled", "canceled", "attrition", "exited", "churn_flag"]}
NUM_F = ["tenure_months", "monthly_fee", "payment_failures", "days_since_last_login", "logins_last_30_days", "usage_minutes", "usage_change_percentage",
         "features_used", "support_tickets", "unresolved_complaints", "customer_satisfaction", "nps", "email_engagement", "campaign_interactions",
         "discount_usage", "days_to_renewal"]
CAT_F = ["contract_type", "subscription_plan"]
FIN_F = ["customer_lifetime_value", "historical_revenue"]
DATE_F = ["subscription_start_date", "renewal_date"]
POST_F = ["cancellation_date"]
NONNEG = ["tenure_months", "monthly_fee", "payment_failures", "days_since_last_login", "logins_last_30_days", "usage_minutes", "features_used",
          "support_tickets", "unresolved_complaints", "campaign_interactions", "days_to_renewal"]
SENSITIVE = {"gender", "sex", "age", "race", "ethnicity", "religion", "nationality", "disability", "marital", "zip", "postcode"}
PRETTY = dict(logins_last_30_days="Logins (30 days)", usage_change_percentage="Usage change %", nps="NPS score", days_since_last_login="Days since last login", monthly_fee="Monthly fee", tenure_stage="Tenure stage")
pretty = lambda f: PRETTY.get(f, str(f).replace("_", " ").capitalize())

# ------------------------------ rendering helpers ------------------------------
def pill(c): return "<span class='pill' style='background:%s26;color:%s'>%s</span>" % (RC.get(c, C["mu"]), RC.get(c, C["mu"]), esc(c))
def mcol(v): return "—" if nul(v) else "<span style='color:%s'>%s</span>" % (C["mi"] if v >= 0 else C["co"], money(v))
def note(t, k="info"):
    col = {"info": C["te"], "ok": C["mi"], "warn": C["am"], "err": C["co"]}[k]
    return "<div class='note' style='border-left:4px solid %s'>%s</div>" % (col, t)
def card(ic, lab, val, desc, col=C["te"]):
    return "<div class='kc' style='border-top:3px solid %s'><div class='ki'>%s</div><div class='kl'>%s</div><div class='kv'>%s</div><div class='kd'>%s</div></div>" % (col, ic, lab, val, desc)
def tbl(d, cols, fmt=None):
    fmt = fmt or {}
    h = "<div class='tw'><table><tr>" + "".join("<th>%s</th>" % n for _, n in cols) + "</tr>"
    for _, r in d.iterrows():
        h += "<tr>"
        for k, _n in cols:
            v = r[k]
            s = fmt[k](v) if k in fmt else ("—" if nul(v) else esc(v))
            h += "<td>%s</td>" % s
        h += "</tr>"
    return h + "</table></div>"
def sty(f, h=380, t=""):
    f.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=C["tx"]), height=h, title=dict(text=t, x=.02),
                    margin=dict(l=50, r=20, t=55, b=50), legend=dict(orientation="h", y=-.25), colorway=[C["te"], C["co"], C["am"], C["mi"]])
    return f
def empty(msg):
    f = go.Figure(); f.add_annotation(text=msg, showarrow=False, font=dict(color=C["mu"], size=14)); f.update_xaxes(visible=False); f.update_yaxes(visible=False)
    return sty(f, 300)
def wilson(k, n, z=1.96):
    if n <= 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); a = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - a) / d, (c + a) / d)
def parse_target(s):
    m = {"1": 1, "0": 0, "yes": 1, "no": 0, "y": 1, "n": 0, "true": 1, "false": 0, "churned": 1, "churn": 1, "cancelled": 1, "canceled": 1, "active": 0, "retained": 0, "stayed": 0, "left": 1}
    out = s.astype(str).str.strip().str.lower().map(m)
    num = pd.to_numeric(s, errors="coerce")
    return out.fillna(num.where(num.isin([0, 1]))).astype(float)
def read_csv_any(file):
    path = file if isinstance(file, str) else getattr(file, "name", file)
    try: df = pd.read_csv(path)
    except Exception: df = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
    if df.empty: raise ValueError("The file contains no rows.")
    if len(df) > 200000: raise ValueError("File too large (limit 200,000 rows).")
    return df
def psi(ref, cur, bins=10):
    ref = np.asarray(ref, float); cur = np.asarray(cur, float); ref = ref[~np.isnan(ref)]; cur = cur[~np.isnan(cur)]
    if len(ref) < 30 or len(cur) < 30: return np.nan
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3: return np.nan
    qs[0] = -np.inf; qs[-1] = np.inf
    a = np.clip(np.histogram(ref, qs)[0] / len(ref), 1e-4, None); b = np.clip(np.histogram(cur, qs)[0] / len(cur), 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))
def econ(p, V, e, cost):
    p = np.asarray(p, float); V = np.asarray(V, float); e = np.asarray(e, float) * np.ones_like(p); cost = np.asarray(cost, float) * np.ones_like(p)
    expo = float(np.nansum(p * V)); ben = float(np.nansum(p * e * V)); c = float(np.nansum(cost)); net = ben - c
    return dict(exposure=expo, benefit=ben, cost=c, net=net, roi=(net / c if c > 0 else np.nan))
def value_of(fee, margin, horizon): return fee * margin * horizon

# ------------------------------ FICTIONAL synthetic data ------------------------------
def make_synth(n=4000, seed=42):
    r = np.random.default_rng(seed); sig = lambda z: 1 / (1 + np.exp(-z))
    plan = r.choice(["Basic", "Standard", "Premium"], n, p=[.42, .38, .20]); contract = r.choice(["Monthly", "Annual", "Two-year"], n, p=[.55, .30, .15])
    disc = (r.random(n) < np.where(plan == "Premium", .15, .28)).astype(int)
    fee = np.round(np.select([plan == "Basic", plan == "Standard"], [12.0, 25.0], 45.0) * np.where(disc == 1, .85, 1.0), 2)
    ten = np.where(r.random(n) < .22, r.integers(1, 4, n), np.clip(np.round(r.gamma(2.4, 9, n)), 1, 72)).astype(int)
    eng = np.clip(r.beta(3, 2.2, n) + .002 * np.minimum(ten, 24) - .05 + r.normal(0, .05, n), .02, 1)
    logins = r.poisson(eng * 18); dsll = np.round(np.clip(r.exponential(30 / (logins + 1)), 0, 89)).astype(int); dsll = np.where(logins == 0, np.maximum(dsll, 14), dsll)
    usage = np.round(eng * r.lognormal(5.3, .45, n)).astype(int)
    decl = r.random(n) < (.12 + .25 * (1 - eng)); uchg = np.round(np.where(decl, -r.uniform(25, 85, n), r.normal(3, 16, n)), 1)
    fu = np.clip(np.round(1 + eng * 7 + r.normal(0, 1.3, n)), 1, 10).astype(int)
    pf = np.minimum(r.poisson(.18 + .35 * (contract == "Monthly") + .1 * disc), 5)
    tick = r.poisson(.5 + 1.0 * (1 - eng)); unres = r.binomial(tick, .25)
    csat = np.clip(np.round(r.normal(7.3 - .9 * np.minimum(unres, 3) - .6 * (pf > 0) + 1.5 * (eng - .6), 1.4)), 1, 10)
    nps = np.clip(np.round(csat + r.normal(0, 1.6, n)), 0, 10)
    email = np.round(np.clip(r.beta(2, 3, n) * (.4 + .6 * eng), 0, 1), 3); camp = r.poisson(.2 + 1.6 * eng)
    dtr = np.where(contract == "Monthly", r.integers(1, 31, n), np.where(contract == "Annual", r.integers(1, 366, n), r.integers(1, 731, n)))
    z0 = (.9 * (contract == "Monthly") - .9 * (contract == "Two-year") + 1.1 * (pf >= 2) + .35 * (pf == 1) + .6 * np.minimum(unres, 3) - .17 * (csat - 7)
          - .018 * uchg - 1.5 * (eng - .6) + .02 * np.minimum(dsll, 40) + .8 * (ten <= 3) - .02 * np.minimum(ten, 36) + .45 * disc - .1 * (fu - 5)
          + .3 * ((plan == "Premium") & (eng < .4)) + r.normal(0, .9, n))
    lo, hi = -8.0, 8.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if sig(z0 + mid).mean() > .22: hi = mid
        else: lo = mid
    churn = (r.random(n) < sig(z0 + (lo + hi) / 2)).astype(int)
    obs = pd.Timestamp("2026-06-30"); hist = np.round(fee * ten * r.uniform(.9, 1.0, n), 2)
    df = pd.DataFrame(dict(
        customer_id=["CU-%05d" % x for x in r.permutation(n) + 10000],
        subscription_start_date=[(obs - pd.Timedelta(days=int(t * 30.44))).strftime("%Y-%m-%d") for t in ten], tenure_months=ten, contract_type=contract,
        subscription_plan=plan, monthly_fee=fee, customer_lifetime_value=np.round(hist + fee * np.where(churn == 1, 0, 12), 2), historical_revenue=hist,
        payment_failures=pf, days_since_last_login=dsll, logins_last_30_days=logins, usage_minutes=usage, usage_change_percentage=uchg, features_used=fu,
        support_tickets=tick, unresolved_complaints=unres, customer_satisfaction=csat, nps=nps, email_engagement=email, campaign_interactions=camp,
        discount_usage=disc, renewal_date=[(obs + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in dtr],
        cancellation_date=[(obs + pd.Timedelta(days=int(r.integers(1, 91)))).strftime("%Y-%m-%d") if c else None for c in churn], churned=churn))
    for col, fr in [("customer_satisfaction", .03), ("nps", .02), ("email_engagement", .015)]: df.loc[r.random(n) < fr, col] = np.nan
    return pd.concat([df, df.sample(6, random_state=1)], ignore_index=True)
SAMPLE = make_synth()

# ------------------------------ ingestion & validation ------------------------------
def clean(raw, override="", snap=""):
    W = []; df = raw.copy(); df.columns = [norm(c) for c in df.columns]
    sens = [c for c in df.columns if set(c.split("_")) & SENSITIVE]
    ov = {}
    for p in re.split(r"[,;\n]", override or ""):
        if "=" in p:
            k, v = p.split("=", 1); ov[norm(k)] = norm(v)
    bad = [k + "=" + v for k, v in ov.items() if k not in SYN or v not in df.columns]
    if bad: W.append("Mapping ignored (unknown field or column): " + ", ".join(bad) + ".")
    ren = {}
    for k, al in SYN.items():
        c = ov[k] if ov.get(k) in df.columns else next((a for a in [k] + al if a in df.columns), None)
        if c is None or c in ren: continue
        if c != k and k in df.columns: df = df.drop(columns=[k])
        ren[c] = k
    df = df.rename(columns=ren); df = df.loc[:, ~df.columns.duplicated()]
    for c in NUM_F + FIN_F:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            s0 = df[c]; df[c] = pd.to_numeric(s0.astype(str).str.replace(r"[$,%₹€£\s]", "", regex=True), errors="coerce")
            nb = int((df[c].isna() & s0.notna() & ~s0.astype(str).str.strip().str.lower().isin(BLANK)).sum())
            if nb: W.append("%d non-numeric values in '%s' treated as missing." % (nb, c))
    for c in DATE_F + POST_F:
        if c in df.columns:
            d0 = df[c]
            try: df[c] = pd.to_datetime(d0, errors="coerce", format="mixed")
            except Exception: df[c] = pd.to_datetime(d0, errors="coerce")
            nb = int((df[c].isna() & d0.notna()).sum())
            if nb: W.append("%d unparseable dates in '%s' treated as missing." % (nb, c))
    n0 = len(df); df = df.drop_duplicates()
    if len(df) < n0: W.append("%d exact duplicate rows removed." % (n0 - len(df)))
    if "customer_id" in df.columns:
        cid = df["customer_id"]; badid = (cid.isna() | (cid.astype(str).str.strip() == "")).values
        df["customer_id"] = cid.astype(str).str.strip()
        if badid.any():
            df.loc[badid, "customer_id"] = ["ROW-%05d" % (i + 1) for i in np.where(badid)[0]]; W.append("%d missing customer IDs replaced by pseudonymous row IDs." % int(badid.sum()))
        d = df["customer_id"].duplicated(keep="first")
        if d.any(): W.append("%d rows repeat a customer_id; first occurrence kept." % int(d.sum())); df = df[~d]
    else:
        df.insert(0, "customer_id", ["ROW-%05d" % (i + 1) for i in range(len(df))]); W.append("No customer_id column: pseudonymous row IDs created.")
    df = df.reset_index(drop=True)
    sn = pd.to_datetime(snap, errors="coerce") if (snap or "").strip() else pd.NaT
    if (snap or "").strip() and pd.isna(sn): W.append("Snapshot date not understood and ignored (use YYYY-MM-DD).")
    if ("tenure_months" not in df.columns or df["tenure_months"].isna().all()) and "subscription_start_date" in df.columns:
        if pd.notna(sn): df["tenure_months"] = ((sn - df["subscription_start_date"]).dt.days / 30.44).clip(lower=0); W.append("tenure_months derived from start date and your snapshot date.")
        else: W.append("start date found but tenure not derived: enter a snapshot date.")
    if ("days_to_renewal" not in df.columns or df["days_to_renewal"].isna().all()) and "renewal_date" in df.columns and pd.notna(sn):
        d = (df["renewal_date"] - sn).dt.days; df["days_to_renewal"] = d.where(d >= 0); W.append("days_to_renewal derived from renewal_date and your snapshot date.")
    if "churned" in df.columns:
        y = parse_target(df["churned"]); nb = int(df["churned"].notna().sum() - y.notna().sum())
        if nb: W.append("%d churn labels could not be interpreted (use 0/1, yes/no, churned/active) and are treated as unlabelled." % nb)
    elif "cancellation_date" in df.columns:
        y = df["cancellation_date"].notna().astype(float); W.append("Churn target derived from cancellation_date (filled = churned). That date is NOT used as a predictor.")
    else:
        y = pd.Series(np.nan, index=df.index); W.append("No churn target found: model training is unavailable; descriptive analytics only.")
    df["churned"] = y
    for c in NONNEG:
        if c in df.columns:
            m = df[c] < 0
            if m.any(): df.loc[m, c] = np.nan; W.append("%d negative values in '%s' set to missing." % (int(m.sum()), c))
    if "email_engagement" in df.columns and 1.5 < df["email_engagement"].max() <= 100: df["email_engagement"] = df["email_engagement"] / 100; W.append("email_engagement looked like a percentage; divided by 100.")
    if "tenure_months" in df.columns and (df["tenure_months"] > 600).any(): W.append("%d records with tenure over 50 years look suspicious." % int((df["tenure_months"] > 600).sum()))
    if {"unresolved_complaints", "support_tickets"} <= set(df.columns):
        n = int((df["unresolved_complaints"] > df["support_tickets"]).sum())
        if n: W.append("%d records have more unresolved complaints than tickets (check definitions)." % n)
    if "monthly_fee" in df.columns and (df["monthly_fee"] == 0).any(): W.append("%d customers have a monthly fee of 0 (their revenue exposure is 0)." % int((df["monthly_fee"] == 0).sum()))
    if sens: W.append("Potentially sensitive columns (" + ", ".join(sens) + ") are NOT used as model features.")
    avail = [k for k in SYN if k in df.columns and df[k].notna().any()]
    for k in SYN:
        if k not in df.columns: df[k] = np.nan
    nums, cats, excl = [], [], []
    for f in NUM_F:
        if f in avail:
            mr = df[f].isna().mean()
            if mr > .6: excl.append((f, "%.0f%% missing" % (mr * 100)))
            elif df[f].nunique() < 2: excl.append((f, "constant"))
            else: nums.append(f)
    for f in CAT_F:
        if f in avail:
            nu = df[f].nunique()
            if nu < 2: excl.append((f, "constant"))
            elif nu > 30: excl.append((f, "too many categories"))
            else: cats.append(f)
    for f in nums + cats:
        mr = df[f].isna().mean()
        if mr > 0: W.append("'%s' missing in %.0f%% of rows: imputed inside the model pipeline (never treated as a real zero)." % (f, mr * 100))
    for f, why in excl: W.append("Feature '%s' excluded from modelling (%s)." % (f, why))
    if not (nums + cats): raise ValueError("No supported predictor columns found. Use the mapping box, e.g. payment_failures=failed_pmts, tenure_months=months.")
    assert not (set(nums + cats) & (set(POST_F) | {"churned", "customer_id"} | set(FIN_F))), "leakage guard"
    return df, W, dict(avail=avail, nums=nums, cats=cats, excl=excl, sens=sens)

def prep_X(df, nums, cats):
    X = pd.DataFrame(index=df.index)
    for c in nums: X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in cats: X[c] = df[c].astype(object).where(df[c].notna(), "Unknown").astype(str).str.strip()
    return X

# ------------------------------ ML engine ------------------------------
def make_ohe():
    try: return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)
def zoo_models(nums, cats):
    def pre(scale):
        t = []
        if nums: t.append(("num", Pipeline([("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])), nums))
        if cats: t.append(("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")), ("oh", make_ohe())]), cats))
        return ColumnTransformer(t, remainder="drop")
    return {"Logistic Regression": Pipeline([("pre", pre(True)), ("m", LogisticRegression(max_iter=2000, C=.5, class_weight="balanced"))]),
            "Random Forest": Pipeline([("pre", pre(False)), ("m", RandomForestClassifier(n_estimators=120, min_samples_leaf=8, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=42))]),
            "Gradient Boosting": Pipeline([("pre", pre(False)), ("m", HistGradientBoostingClassifier(max_depth=3, learning_rate=.06, max_iter=150, l2_regularization=1.0, random_state=42))])}
def calibrated(pipe):
    try: return CalibratedClassifierCV(estimator=clone(pipe), method="sigmoid", cv=3)
    except TypeError: return CalibratedClassifierCV(base_estimator=clone(pipe), method="sigmoid", cv=3)
def oof_proba(model, X, y, n=5):
    return cross_val_predict(clone(model), X, y, cv=StratifiedKFold(n_splits=n, shuffle=True, random_state=42), method="predict_proba")[:, 1]
def pick_threshold(y, p, obj, c_fn, c_fp):
    y = np.asarray(y).astype(int); p = np.asarray(p); g = np.linspace(.02, .95, 94); P = (p[:, None] >= g[None, :])
    tp = (P & (y[:, None] == 1)).sum(0).astype(float); fp = (P & (y[:, None] == 0)).sum(0).astype(float); fn = (y == 1).sum() - tp
    if str(obj).startswith("Cost"): score = -(fn * c_fn + fp * c_fp)
    else:
        b2 = 4.0 if str(obj).startswith("Prioritise") else 1.0
        score = (1 + b2) * tp / np.maximum((1 + b2) * tp + b2 * fn + fp, 1e-9)
    return float(g[int(np.argmax(score))])
def mets(y, p, t):
    pred = (p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(roc=roc_auc_score(y, p), pr=average_precision_score(y, p), prec=precision_score(y, pred, zero_division=0), rec=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), brier=brier_score_loss(y, p), thr=t, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
def contributions(model, X, ref):
    base = model.predict_proba(X)[:, 1]; Cm = np.zeros((len(X), X.shape[1]))
    for j, f in enumerate(X.columns):
        Xm = X.copy(); Xm[f] = ref[f]; Cm[:, j] = base - model.predict_proba(Xm)[:, 1]
    return Cm, base
def risk_of(p, cuts):
    p = np.asarray(p, float); cm, ch, cc = cuts
    return np.select([p >= cc, p >= ch, p >= cm, ~np.isnan(p)], ["Critical", "High", "Moderate", "Low"], "Unscored")

def train_all(df, info, cfg):
    nums, cats = info["nums"], info["cats"]
    M = dict(ok=False, why="", warn=[], nums=nums, cats=cats, feats=nums + cats)
    lab = df[df["churned"].notna()]; yv = lab["churned"].astype(int).values
    if len(lab) < 150 or yv.sum() < 30 or (len(yv) - yv.sum()) < 30:
        M["why"] = "training needs at least 150 labelled rows and 30 customers in each class (found %d labelled, %d churned)." % (len(lab), int(yv.sum())); return M
    Xall = prep_X(df, nums, cats); X = Xall.loc[lab.index]
    for f in nums:
        ok = X[f].notna().values
        if ok.sum() > 50 and len(np.unique(yv[ok])) == 2:
            a = roc_auc_score(yv[ok], X[f].values[ok])
            if a > .97 or a < .03: M["warn"].append("'%s' separates churners almost perfectly (AUC %.2f); check for target leakage." % (f, a))
    tr, te = train_test_split(np.arange(len(lab)), test_size=.25, stratify=yv, random_state=42)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], yv[tr], yv[te]; R = {}
    for name, pipe in zoo_models(nums, cats).items():
        try:
            oof = oof_proba(pipe, Xtr, ytr); m = clone(pipe).fit(Xtr, ytr)
            R[name] = dict(pipe=pipe, model=m, p=m.predict_proba(Xte)[:, 1], cv_auc=float(roc_auc_score(ytr, oof)), thr=pick_threshold(ytr, oof, cfg["obj"], cfg["c_fn"], cfg["c_fp"]))
        except Exception as e: M["warn"].append("%s could not be trained (%s)." % (name, type(e).__name__))
    if not R: M["why"] = "all candidate models failed to train."; return M
    for r_ in R.values(): r_["test"] = mets(yte, r_["p"], r_["thr"])
    best = max(R, key=lambda k: R[k]["cv_auc"])
    cal = calibrated(R[best]["pipe"]); oof_c = oof_proba(cal, Xtr, ytr); thr = pick_threshold(ytr, oof_c, cfg["obj"], cfg["c_fn"], cfg["c_fp"])
    pc = clone(cal).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    final = clone(cal).fit(X, yv); oof_all = oof_proba(cal, X, yv); imp = None
    try:
        pi = permutation_importance(R[best]["model"], Xte, yte, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=1)
        imp = pd.DataFrame(dict(feature=list(Xte.columns), imp=pi.importances_mean, sd=pi.importances_std)).sort_values("imp", ascending=False)
    except Exception: pass
    rng = np.random.default_rng(0); bs = []
    for _ in range(300):
        i = rng.integers(0, len(yte), len(yte))
        if yte[i].min() != yte[i].max(): bs.append(roc_auc_score(yte[i], pc[i]))
    ref = {f: float(X[f].median()) for f in nums}; ref.update({f: X[f].mode().iat[0] for f in cats})
    M.update(ok=True, R=R, best=best, thr=thr, cal_p=pc, yte=yte, Xte=Xte, final=final, oof_all=oof_all, ref=ref, imp=imp,
             auc_ci=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (np.nan, np.nan), n_train=len(tr), n_test=len(te),
             test_cal=mets(yte, pc, thr), cv_cal=float(roc_auc_score(ytr, oof_c)), refnum={f: X[f].values for f in nums})
    return M

# ------------------------------ scoring, segments, strategies ------------------------------
SEG_ORDER = ["High-value, elevated risk", "Payment friction", "Unresolved support issues", "Onboarding struggle", "Declining engagement", "Renewal at risk",
             "Price-sensitive", "Low engagement / adoption", "Stable & loyal"]
ALLSEG = SEG_ORDER + ["General monitoring"]
SEG2STRAT = {"High-value, elevated risk": "High-value outreach", "Payment friction": "Payment recovery", "Unresolved support issues": "Service recovery",
             "Onboarding struggle": "Onboarding assistance", "Declining engagement": "Re-engagement campaign", "Renewal at risk": "Renewal check-in",
             "Price-sensitive": "Plan adjustment", "Low engagement / adoption": "Feature discovery", "Stable & loyal": "Monitor only", "General monitoring": "Monitor only"}
STRAT_COLS = ["Strategy", "Cost per customer ($)", "Assumed effectiveness"]
DEFAULT_STRATS = pd.DataFrame([["Onboarding assistance", 10, .15], ["Payment recovery", 3, .25], ["Service recovery", 25, .15], ["Re-engagement campaign", 4, .10],
                               ["Plan adjustment", 6, .12], ["High-value outreach", 40, .20], ["Feature discovery", 5, .08], ["Renewal check-in", 5, .08], ["Monitor only", 0, 0.0]], columns=STRAT_COLS)
PLAY = {
 "High-value, elevated risk": ("High-value account with elevated modeled risk; the data cannot say why.", "Proactive account-owner outreach: ask about goals and friction; tailor the plan (no blanket discount).", "Personal email or call from a named account owner", "Within 3 business days"),
 "Payment friction": ("Repeated failed payments often reflect expired cards or bank holds (involuntary churn), not intent to leave.", "Respectful payment-recovery message with an easy way to update details or change billing date.", "Email plus in-app notice", "Right after a failed payment; one follow-up after about 5 days"),
 "Unresolved support issues": ("Repeated unresolved issues may be linked to dissatisfaction (association, not proof).", "Service recovery: escalate to a human agent, acknowledge the issue, confirm resolution.", "Phone or personal email from the support lead", "Within 2 business days"),
 "Onboarding struggle": ("A new customer with weak early adoption may not have reached first value.", "Guided onboarding: short walkthrough and proactive help with the core feature.", "In-app guide plus email", "Within the first weeks of tenure"),
 "Declining engagement": ("Falling usage may signal lost interest or changed needs (hypothesis).", "Personalised re-engagement: highlight relevant features and ask for feedback.", "Email or in-app message", "Within 1 week"),
 "Renewal at risk": ("Weak engagement close to renewal raises the chance of non-renewal.", "Timely renewal check-in: summarise value used and offer plan options.", "Email", "2-4 weeks before renewal"),
 "Price-sensitive": ("Past discount use may indicate price sensitivity (hypothesis).", "Transparent plan comparison and a suitable lower-cost plan; no pressure or fake urgency.", "Email", "Before the next billing date"),
 "Low engagement / adoption": ("Few features used and few logins suggest limited adoption.", "Relevant tutorial or feature recommendation based on the plan.", "Email or in-app", "Within 2 weeks"),
 "Stable & loyal": ("No warning signals in the supplied fields.", "No outreach; keep monitoring.", "None", "Re-check next scoring cycle"),
 "General monitoring": ("No specific rule matched.", "Keep monitoring; no targeted action recommended.", "None", "Next scoring cycle")}
MSG = {
 "Payment friction": "Subject: A quick note about your {plan} payment\n\nHello,\n\nA recent payment for your {plan} subscription did not go through. This can happen with an expired card or a bank hold. You can update your payment details here: [secure link]. If another billing date or method suits you better, just reply and we will help.\n\nThank you,\n[Team name]",
 "Onboarding struggle": "Subject: Getting started with your {plan} plan\n\nHello,\n\nWelcome. We would love to help you get the most from your {plan} plan. Here is a short guide: [guide link]. If you prefer, reply and we can arrange a quick walkthrough.\n\nBest wishes,\n[Team name]",
 "Declining engagement": "Subject: Is {plan} still working for you?\n\nHello,\n\nWe noticed you have used {plan} less recently. If something is not working or your needs have changed, we would genuinely like to hear it. Here are a few features that may help: [feature link].\n\nBest wishes,\n[Team name]",
 "Unresolved support issues": "Subject: Following up on your recent support requests\n\nHello,\n\nWe are sorry that your recent support request(s) have not been fully resolved. A member of our team has been asked to review them personally. Reply to this message or call [number] and we will pick it up straight away.\n\nSincerely,\n[Support lead name]",
 "Price-sensitive": "Subject: Making sure your plan fits\n\nHello,\n\nHere is a clear comparison of our plans so you can choose what fits you best: [comparison link]. No action is needed if you are happy with {plan}.\n\nBest wishes,\n[Team name]",
 "High-value, elevated risk": "Subject: How are things going with {plan}?\n\nHello,\n\nAs a valued {plan} customer, we would like to hear how things are going and whether we can do anything better. Would you have 15 minutes for a call with [account owner name]? Suggested times: [times].\n\nBest wishes,\n[Account owner name]",
 "Low engagement / adoption": "Subject: Features that may help you\n\nHello,\n\nMany {plan} customers find these features useful: [feature links]. A two-minute tutorial is here: [tutorial link]. Reply if you would like help setting anything up.\n\nBest wishes,\n[Team name]",
 "Renewal at risk": "Subject: Your {plan} renewal{when}\n\nHello,\n\nYour subscription renews soon. We wanted to check that it is still meeting your needs and show you the options available, including changing or pausing your plan: [plan options link].\n\nBest wishes,\n[Team name]"}

def risk_cuts(cfg, M, S):
    if cfg["auto"] and M.get("ok"):
        base = float(S["churned"].mean()); high = float(np.clip(M["thr"], .10, .80)); mod = max(.05, min(base, high - .05)); crit = min(.90, max(.60, high + .15)); c = [mod, high, crit]
    else: c = [float(x) for x in cfg["t"]]
    return [round(x, 2) for x in sorted(c)]

def score_all(df, M, cfg):
    S = df.copy(); nums, cats = M["nums"], M["cats"]
    S["active"] = (S["churned"].fillna(0).astype(int) == 0); S["p"] = np.nan; S["drivers"] = ""
    Xall = prep_X(S, nums, cats)
    if M["ok"]:
        lab = S["churned"].notna().values; p = np.full(len(S), np.nan); p[lab] = M["oof_all"]
        if (~lab).any(): p[~lab] = M["final"].predict_proba(Xall[~lab])[:, 1]
        S["p"] = p
    S["value"] = value_of(S["monthly_fee"], cfg["margin"], cfg["horizon"]); S["mrr_exp"] = S["p"] * S["monthly_fee"]; S["exp_contrib"] = S["p"] * S["value"]
    S["n_missing"] = Xall[nums].isna().sum(axis=1) if nums else 0; S["completeness"] = 1 - S["n_missing"] / max(len(nums), 1)
    S["tenure_stage"] = pd.cut(S["tenure_months"], [-0.01, 3, 6, 12, 24, np.inf], labels=["0-3 mo", "4-6 mo", "7-12 mo", "13-24 mo", "25+ mo"])
    cuts = risk_cuts(cfg, M, S); G["cuts"] = cuts; S["risk"] = risk_of(S["p"].values, cuts)
    if M["ok"]:
        idx = np.where(S["active"].values & S["p"].notna().values)[0]
        if len(idx):
            Xa = Xall.iloc[idx]; Cm, _ = contributions(M["final"], Xa, M["ref"]); names = [pretty(f) for f in Xa.columns]; out = []
            for row in Cm:
                parts = ["%s (+%.0f pts)" % (names[j], row[j] * 100) for j in np.argsort(-row)[:3] if row[j] >= .01]
                out.append("; ".join(parts) if parts else "No dominant driver")
            S.loc[S.index[idx], "drivers"] = out
    return S

def seg_flags(S, c):
    q = lambda col: S[col].quantile(.25) if S[col].notna().any() else np.nan
    low_log = S["logins_last_30_days"] <= q("logins_last_30_days"); low_feat = S["features_used"] <= q("features_used")
    F = pd.DataFrame(index=S.index); not_new = ~(S["tenure_months"] <= c["onb"]); qfee = S["monthly_fee"].quantile(c["hvq"]) if S["monthly_fee"].notna().any() else np.nan
    F["High-value, elevated risk"] = (S["monthly_fee"] >= qfee) & S["risk"].isin(["High", "Critical"])
    F["Payment friction"] = S["payment_failures"] >= c["pf"]
    F["Unresolved support issues"] = S["unresolved_complaints"] >= c["unr"]
    F["Onboarding struggle"] = (S["tenure_months"] <= c["onb"]) & (low_log | low_feat)
    F["Declining engagement"] = (S["usage_change_percentage"] <= -c["decl"]) & not_new
    F["Renewal at risk"] = (S["days_to_renewal"] <= 30) & (low_log | low_feat | (S["usage_change_percentage"] <= -c["decl"] / 2))
    F["Price-sensitive"] = S["discount_usage"] > 0
    F["Low engagement / adoption"] = low_log & low_feat
    others = F.fillna(False).astype(bool).any(axis=1); F["Stable & loyal"] = (~others) & (S["tenure_months"] >= 12) & S["risk"].isin(["Low", "Unscored"])
    G["q"] = dict(logins=q("logins_last_30_days"), feats=q("features_used"))
    return F.fillna(False).astype(bool)

def assign_strategy():
    S = G["S"]; sd = G["strat"].set_index("Strategy")
    S["strategy"] = S["segment"].map(SEG2STRAT).fillna("Monitor only")
    S["cost"] = S["strategy"].map(sd["Cost per customer ($)"]).astype(float); S["eff"] = S["strategy"].map(sd["Assumed effectiveness"]).astype(float)
    S["benefit"] = S["p"] * S["eff"] * S["value"]; S["net"] = S["benefit"] - S["cost"]; S["prio_rank"] = np.nan
    act = S[S["active"] & S["p"].notna()].sort_values(["net", "p"], ascending=False, na_position="last")
    S.loc[act.index, "prio_rank"] = np.arange(1, len(act) + 1)

def apply_segments():
    S = G["S"]; F = seg_flags(S, G["segcfg"]); G["F"] = F; prim = pd.Series("General monitoring", index=S.index)
    for nme in reversed(SEG_ORDER): prim[F[nme]] = nme
    S["segment"] = prim; assign_strategy()

def evidence(r):
    ev = []; has = lambda c: pd.notna(r.get(c)); q = G.get("q", {})
    if has("payment_failures") and r["payment_failures"] >= 1: ev.append("%d failed payment(s)" % r["payment_failures"])
    if has("unresolved_complaints") and r["unresolved_complaints"] >= 1: ev.append("%d unresolved complaint(s)" % r["unresolved_complaints"])
    if has("usage_change_percentage") and r["usage_change_percentage"] <= -15: ev.append("usage %+.0f%% vs prior period" % r["usage_change_percentage"])
    if has("tenure_months") and r["tenure_months"] <= 3: ev.append("only %.0f month(s) of tenure" % r["tenure_months"])
    if has("features_used") and pd.notna(q.get("feats")) and r["features_used"] <= q["feats"]: ev.append("%.0f feature(s) used (bottom quartile)" % r["features_used"])
    if has("logins_last_30_days") and pd.notna(q.get("logins")) and r["logins_last_30_days"] <= q["logins"]: ev.append("%.0f logins in 30 days (bottom quartile)" % r["logins_last_30_days"])
    if has("days_since_last_login") and r["days_since_last_login"] >= 14: ev.append("%.0f days since last login" % r["days_since_last_login"])
    if has("discount_usage") and r["discount_usage"] > 0: ev.append("has used a discount")
    if has("days_to_renewal") and r["days_to_renewal"] <= 30: ev.append("renewal in %.0f days" % r["days_to_renewal"])
    if has("customer_satisfaction") and r["customer_satisfaction"] <= 5: ev.append("satisfaction score %.0f" % r["customer_satisfaction"])
    return ev or ["no strong single warning signal in the supplied fields"]

def rec_card(r):
    hyp, act, ch, tm = PLAY[r["segment"]]; conf = []
    if pd.notna(r["completeness"]) and r["completeness"] < .75: conf.append("%d predictor(s) missing" % r["n_missing"])
    if pd.notna(r["p"]) and any(abs(r["p"] - c) < .05 for c in G["cuts"]): conf.append("score is near a risk cut-off")
    conf.append("effectiveness is an assumption; validate with an A/B test")
    if pd.isna(r["value"]): conf.append("monthly fee missing: no value estimate")
    rows = [("1. Customer / segment", "%s · %s" % (esc(r["customer_id"]), esc(r["segment"]))), ("2. Risk", "%s %s (calibrated model probability)" % (pill(r["risk"]), pc0(r["p"]))),
            ("3. Observed evidence", esc("; ".join(evidence(r)))), ("4. Possible explanation", esc(hyp) + " <i>(hypothesis, not a verified reason)</i>"),
            ("5. Recommended action", "<b>%s</b>: %s" % (esc(r["strategy"]), esc(act))), ("6. Channel", esc(ch) + " <i>(contact preference not in data)</i>"), ("7. Timing", esc(tm)),
            ("8. Estimated cost", money(r["cost"]) + " per customer (editable in tab G)"),
            ("9. Expected value (hypothetical)", (money(r["benefit"]) + " = risk × assumed effectiveness × value") if pd.notna(r["benefit"]) else "—"), ("10. Confidence & caveats", esc("; ".join(conf)))]
    return "<div class='tw'><table>" + "".join("<tr><th style='width:230px'>%s</th><td>%s</td></tr>" % (a, b) for a, b in rows) + "</table></div>"

def draft_msg(r):
    t = MSG.get(r["segment"])
    if not t: return "No outreach recommended for this customer (monitor only)."
    plan = str(r["subscription_plan"]) if pd.notna(r["subscription_plan"]) else "subscription"
    when = " in %.0f days" % r["days_to_renewal"] if pd.notna(r.get("days_to_renewal")) else ""
    return t.format(plan=plan, when=when) + "\n\n[Draft only. Review and edit before use. Nothing is sent automatically. Add no discount unless approved.]"

# ------------------------------ pipeline entry ------------------------------
def analyze(file, mapping, snap, margin, horizon, c_fp, c_fn, obj, auto, t1, t2, t3):
    G.clear()
    try:
        demo = file is None; raw = SAMPLE.copy() if demo else read_csv_any(file)
        cfg = dict(margin=float(margin or .6), horizon=int(horizon or 12), c_fp=float(c_fp if c_fp is not None else 12), obj=obj or OBJS[0], auto=bool(auto), t=(t1, t2, t3), mapping=mapping or "", snap=snap or "", demo=demo)
        df, W, info = clean(raw, mapping, snap)
        v = value_of(df["monthly_fee"], cfg["margin"], cfg["horizon"]).median()
        if c_fn is not None and not pd.isna(c_fn) and float(c_fn) > 0: cfg["c_fn"] = float(c_fn)
        elif pd.notna(v): cfg["c_fn"] = float(v); W.append("Cost of a missed churner set to the median customer value (%s) because none was supplied." % money(v))
        else: cfg["c_fn"] = 100.0; W.append("No fee data and no missed-churner cost: 100 used for the cost-based threshold only.")
        M = train_all(df, info, cfg); W += M["warn"]
        G.update(df=df, info=info, W=W, M=M, cfg=cfg, strat=DEFAULT_STRATS.copy(), segcfg=dict(onb=3, decl=30, pf=2, unr=2, hvq=.75), raw_n=len(raw), demo=demo)
        G["S"] = score_all(df, M, cfg); apply_segments()
        msg = ("Fictional demo data analysed. " if demo else "Your data analysed. ") + "%d customers, %d labelled. " % (len(df), df["churned"].notna().sum())
        msg += ("Best model: %s (calibrated), held-out ROC-AUC %.3f." % (M["best"], M["test_cal"]["roc"])) if M["ok"] else "Model unavailable: " + M["why"]
        return note(esc(msg), "ok" if M["ok"] else "warn")
    except Exception as e:
        G.clear(); return note("❌ " + esc(e), "err")

def refresh():
    if "S" not in G: return NA(8)
    S = G["S"]; M = G["M"]
    plans = sorted(S["subscription_plan"].dropna().astype(str).unique().tolist()); cons = sorted(S["contract_type"].dropna().astype(str).unique().tolist())
    ids = S[S["active"] & S["p"].notna()].sort_values("prio_rank").head(300)["customer_id"].tolist()
    dims = (["tenure_stage"] if S["tenure_months"].notna().any() else []) + M["cats"] + M["nums"]
    return (gr.update(choices=["All"] + plans, value="All"), gr.update(choices=["All"] + ALLSEG, value="All"), gr.update(choices=ids, value=ids[0] if ids else None),
            gr.update(choices=ids, value=ids[0] if ids else None), gr.update(choices=plans), gr.update(choices=cons),
            gr.update(choices=dims, value=dims[0] if dims else None), gr.update(value=DEFAULT_STRATS.copy()))

# ------------------------------ A: executive overview ------------------------------
def effects(L, nums):
    rows = []; k = max(len(nums), 1)
    for f in nums:
        a = L.loc[L["churned"] == 0, f].dropna(); b = L.loc[L["churned"] == 1, f].dropna()
        if len(a) < 30 or len(b) < 30: continue
        u, pv = stats.mannwhitneyu(b, a, alternative="two-sided"); rows.append(dict(feature=f, n_ret=len(a), n_ch=len(b), mean_ret=a.mean(), mean_ch=b.mean(), auc=u / (len(a) * len(b)), p=pv, p_adj=min(1.0, pv * k)))
    d = pd.DataFrame(rows)
    if len(d): d = d.assign(_s=(d["auc"] - .5).abs()).sort_values("_s", ascending=False).drop(columns="_s")
    return d

def render_exec():
    if "S" not in G: return NA(5)
    S = G["S"]; M = G["M"]; cfg = G["cfg"]; A = S[S["active"]]; lab = S[S["churned"].notna()]; sc = A[A["p"].notna()]
    obs = lab["churned"].mean() if len(lab) else np.nan; hr = sc[sc["risk"].isin(["High", "Critical"])]; ok = M["ok"]
    win = "Illustrative synthetic label: cancelled within 90 days of the 2026-06-30 snapshot" if cfg["demo"] else "As defined by the churn column in your file"
    cards = [card("👥", "Active customers", "{:,}".format(len(A)), "of {:,} customers in the file".format(len(S))), card("📉", "Observed churn · historical", pc1(obs), win, C["co"]),
             card("🎯", "Predicted high-risk · model", "{:,}".format(len(hr)) if ok else "—", ("%.1f%% of scored active customers (High + Critical)" % (100 * len(hr) / max(len(sc), 1))) if ok else "model unavailable", C["am"]),
             card("💸", "MRR in high-risk accounts", money(hr["monthly_fee"].sum(min_count=1)) if ok else "—", "Sum of monthly fees (customers with a fee on file)", C["am"]),
             card("⏳", "Expected MRR at risk", money(sc["mrr_exp"].sum(min_count=1)) if ok else "—", "Σ risk × monthly fee (probability-weighted)", C["co"]),
             card("🧮", "Contribution exposure", money(sc["exp_contrib"].sum(min_count=1)) if ok else "—", "Σ risk × fee × %.0f%% margin × %d mo · gross, not recoverable" % (cfg["margin"] * 100, cfg["horizon"]), C["mi"]),
             card("🧠", "Held-out ROC-AUC", fx(M["test_cal"]["roc"], 3) if ok else "—", ("95%% CI %s–%s · synthetic data ≠ real-world accuracy" % (fx(M["auc_ci"][0]), fx(M["auc_ci"][1]))) if ok else "", C["te"]),
             card("🔬", "Held-out PR-AUC", fx(M["test_cal"]["pr"], 3) if ok else "—", ("churn prevalence in test set %.1f%%" % (100 * M["yte"].mean())) if ok else "", C["mi"])]
    k = "<div class='kg'>" + "".join(cards) + "</div>"
    if cfg["demo"]: k = note("<b>ILLUSTRATIVE SYNTHETIC DATA</b>: fictional customers for demonstration. Nothing here describes real customers.", "warn") + k
    k += note("<b>Historical</b> = observed churn labels in the file. <b>Predicted</b> = model risk for customers who have not churned. Scores show association, not guarantees.", "info")
    miss = A["monthly_fee"].isna().mean()
    if miss > 0: k += note("monthly_fee is missing for %.0f%% of active customers; they are excluded from money metrics (never counted as zero)." % (miss * 100), "warn")
    cnt = sc["risk"].value_counts().reindex(RISKS).fillna(0)
    f1 = sty(go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=[RC[c] for c in cnt.index], text=cnt.values.astype(int), textposition="outside")), 340, "Active customers by predicted risk") if ok else empty("Model unavailable")
    sg = S.groupby("segment").agg(n_act=("active", "sum"), obs=("churned", "mean"), nl=("churned", "count")).reset_index().sort_values("n_act", ascending=False)
    f2 = go.Figure([go.Bar(x=sg["segment"], y=sg["n_act"], name="Active customers", marker_color=C["te"]),
                    go.Scatter(x=sg["segment"], y=sg["obs"], name="Observed churn rate", yaxis="y2", mode="markers+lines", marker=dict(color=C["co"], size=9))])
    f2 = sty(f2, 340, "Segment mix and observed churn rate"); f2.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False), xaxis=dict(tickangle=-25))
    eff = effects(lab, M["nums"]) if len(lab) >= 60 else pd.DataFrame()
    if len(eff):
        e = eff.head(8).iloc[::-1]; f3 = go.Figure(go.Bar(y=[pretty(x) for x in e["feature"]], x=e["auc"] - .5, orientation="h", marker_color=[C["co"] if v > 0 else C["te"] for v in e["auc"] - .5]))
        f3 = sty(f3, 340, "Behaviours associated with churn (AUC − 0.5; right = higher in churners)")
    else: f3 = empty("Needs labelled churn data")
    if ok:
        g = hr.groupby("segment").agg(n=("p", "size"), p=("p", "mean"), ben=("benefit", "sum"), cost=("cost", "sum")).reset_index(); g["net"] = g["ben"] - g["cost"]
        g = g[g["segment"] != "Stable & loyal"].sort_values("net", ascending=False).head(5)
        pr = "<div class='sec'>Recommended retention priorities (hypothetical, among High + Critical customers)</div>" + tbl(g, [("segment", "Segment"), ("n", "Customers"), ("p", "Avg risk"), ("ben", "Assumed benefit"), ("cost", "Cost"), ("net", "Net")], dict(n=n0f, p=pc0, ben=money, cost=money, net=mcol))
        top = sc.sort_values("prio_rank").head(5)
        pr += "<div class='sec'>Top 5 customers by expected net benefit of action</div>" + tbl(top, [("customer_id", "Customer"), ("p", "Risk"), ("segment", "Segment"), ("strategy", "Suggested action"), ("net", "Est. net")], dict(p=pc0, net=mcol))
        pr += note("Priority = expected net benefit (risk × assumed effectiveness × value − cost), not risk alone. Effectiveness values are assumptions; confirm with controlled experiments.", "info")
    else: pr = note("Model unavailable: " + esc(M["why"]), "warn")
    return k, f1, f2, f3, pr

# ------------------------------ B: risk intelligence ------------------------------
def render_risk(cats, plan, seg, tmin, tmax, q, sortby, topn):
    if "S" not in G: return note("Run the analysis first.", "info"), None, None
    S = G["S"]; M = G["M"]
    if not M["ok"]: return note("Risk scoring unavailable: " + esc(M["why"]), "warn"), pd.DataFrame(), None
    A = S[S["active"] & S["p"].notna()].copy(); n_all = len(A)
    if cats: A = A[A["risk"].isin(cats)]
    if plan and plan != "All": A = A[A["subscription_plan"].astype(str) == plan]
    if seg and seg != "All": A = A[A["segment"] == seg]
    if tmin is not None: A = A[A["tenure_months"] >= tmin]
    if tmax is not None: A = A[A["tenure_months"] <= tmax]
    if q: A = A[A["customer_id"].str.contains(str(q).strip(), case=False, regex=False)]
    key = {"Priority (expected net benefit)": ("prio_rank", True), "Risk score": ("p", False), "Monthly fee": ("monthly_fee", False), "Expected MRR exposure": ("mrr_exp", False)}.get(sortby, ("prio_rank", True))
    A = A.sort_values(key[0], ascending=key[1], na_position="last"); tot = len(A); A = A.head(int(topn or 100)); cm, ch, cc = G["cuts"]; notes = []
    for r in A.itertuples():
        x = []
        if r.completeness < .75: x.append("%d predictor(s) missing" % r.n_missing)
        if min(abs(r.p - c) for c in G["cuts"]) < .05: x.append("near a cut-off")
        notes.append("; ".join(x))
    out = pd.DataFrame({"Priority #": A["prio_rank"].values, "Customer ID": A["customer_id"].values, "Risk score %": (A["p"] * 100).round(1).values, "Risk": A["risk"].map(BADGE).values,
                        "Segment": A["segment"].values, "Plan": A["subscription_plan"].values, "Tenure (mo)": A["tenure_months"].round(0).values, "Monthly fee": A["monthly_fee"].round(2).values,
                        "Expected MRR exposure": A["mrr_exp"].round(2).values, "Est. net benefit": A["net"].round(0).values, "Main model drivers": A["drivers"].values,
                        "Suggested next action": A["strategy"].values, "Data note": notes})
    path = None
    if len(out): path = "churn_ai_risk_list.csv"; out.to_csv(path, index=False)
    info = note("Showing <b>%d</b> of %d matching active customers (%d scored). Cut-offs: Moderate ≥ %.0f%%, High ≥ %.0f%%, Critical ≥ %.0f%% (%s). Scores are calibrated out-of-fold probabilities. "
                "Priority = expected net benefit of the suggested action, not risk alone. Copy a Customer ID into tab E for the full profile." %
                (len(out), tot, n_all, cm * 100, ch * 100, cc * 100, "auto: moderate = base rate, high = decision threshold" if G["cfg"]["auto"] else "manual"), "info")
    return info, out, path

def batch_predict(file):
    if "M" not in G or not G["M"]["ok"]: return note("Train a model first (run the analysis on labelled data).", "warn"), None, None
    if file is None: return note("Upload a CSV of customers to score.", "info"), None, None
    try:
        raw = read_csv_any(file); df, W, info = clean(raw, G["cfg"]["mapping"], G["cfg"]["snap"]); M = G["M"]; X = prep_X(df, M["nums"], M["cats"])
        miss = [f for f in M["feats"] if df[f].isna().all()]; p = M["final"].predict_proba(X)[:, 1]
        out = pd.DataFrame({"Customer ID": df["customer_id"], "Risk score %": (p * 100).round(1), "Risk": [BADGE[c] for c in risk_of(p, G["cuts"])], "Predictors missing": X[M["nums"]].isna().sum(axis=1).values})
        if len(df) <= 5000:
            Cm, _ = contributions(M["final"], X, M["ref"]); nm = [pretty(f) for f in X.columns]
            out["Main model drivers"] = ["; ".join("%s (+%.0f pts)" % (nm[j], row[j] * 100) for j in np.argsort(-row)[:3] if row[j] >= .01) or "No dominant driver" for row in Cm]
        path = "churn_ai_batch_scores.csv"; out.to_csv(path, index=False)
        ps = pd.DataFrame([dict(f=pretty(f), psi=psi(M["refnum"][f], df[f].values)) for f in M["nums"]]); ps["status"] = np.where(ps["psi"] > .25, "major shift", np.where(ps["psi"] > .10, "moderate shift", "stable"))
        h = note("Scored %d customers with the final calibrated model. Customers that were in the training data get optimistic in-sample scores." % len(out), "ok")
        for w in W[:6]: h += note(esc(w), "warn")
        if miss: h += note("Predictors entirely missing in this file (imputed): " + esc(", ".join(pretty(f) for f in miss)), "warn")
        h += "<div class='sec'>Input drift vs training data (PSI)</div>" + tbl(ps, [("f", "Feature"), ("psi", "PSI"), ("status", "Status")], dict(psi=lambda v: fx(v, 3)))
        h += note("PSI < 0.10 stable, 0.10–0.25 moderate, > 0.25 major shift. This checks inputs only; performance monitoring needs future labelled snapshots and is not implemented.", "info")
        return h, out.head(500), path
    except Exception as e: return note("❌ " + esc(e), "err"), None, None

# ------------------------------ C: behavioural patterns ------------------------------
def churn_by(L, f):
    s = L[f]
    if f == "tenure_tage" or f == "tenure_stage": key = s
    elif f in G["info"]["cats"]: key = s.astype(str)
    elif s.nunique() <= 8: key = s
    else: key = pd.qcut(s, 5, duplicates="drop")
    d = L.assign(_k=key).dropna(subset=["_k"]).groupby("_k", observed=True)["churned"].agg(["sum", "count"]).reset_index()
    d["rate"] = d["sum"] / d["count"]; ci = [wilson(a, b) for a, b in zip(d["sum"], d["count"])]; d["lo"] = [c[0] for c in ci]; d["hi"] = [c[1] for c in ci]; d["label"] = d["_k"].astype(str)
    return d

def pat_chart(f):
    if "S" not in G or not f: return empty("Run the analysis first")
    L = G["S"][G["S"]["churned"].notna()]
    if len(L) < 60 or f not in L.columns: return empty("Needs labelled churn data")
    d = churn_by(L, f); ttl = "Observed churn rate by %s (95%% Wilson interval)" % pretty(f).lower()
    if f in G["info"]["cats"]:
        try: ttl += " · chi-square p = %.3g" % stats.chi2_contingency(pd.crosstab(L[f].astype(str), L["churned"]))[1]
        except Exception: pass
    fig = go.Figure(go.Bar(x=d["label"], y=d["rate"], marker_color=C["te"], text=["n=%d" % n for n in d["count"]], textposition="outside",
                           error_y=dict(type="data", symmetric=False, array=(d["hi"] - d["rate"]).values, arrayminus=(d["rate"] - d["lo"]).values, color=C["mu"])))
    fig = sty(fig, 400, ttl); fig.update_yaxes(tickformat=".0%"); return fig

def render_patterns():
    if "S" not in G: return NA(4)
    S = G["S"]; L = S[S["churned"].notna()]
    if len(L) < 60:
        e = empty("Needs a labelled churn column"); return e, note("Pattern analysis needs at least 60 labelled customers.", "warn"), e, e
    first = "tenure_stage" if S["tenure_months"].notna().any() else (G["info"]["cats"] + G["info"]["nums"])[0]
    eff = effects(L, G["M"]["nums"]); h = ""
    if len(eff):
        eff["diff"] = (eff["mean_ch"] - eff["mean_ret"]) / eff["mean_ret"].abs().replace(0, np.nan); eff["dir"] = np.where(eff["auc"] > .5, "higher in churners", "lower in churners"); eff["sig"] = np.where(eff["p_adj"] < .05, "yes", "no")
        h = "<div class='sec'>Retained vs churned: numeric behaviours</div>" + tbl(eff, [("feature", "Feature"), ("mean_ret", "Mean (retained)"), ("mean_ch", "Mean (churned)"), ("diff", "Difference"), ("dir", "Direction"), ("auc", "Separability (AUC)"),
                                                                                         ("p_adj", "p (Bonferroni)"), ("sig", "Significant"), ("n_ch", "n churned")],
                                                                                   dict(feature=lambda v: esc(pretty(v)), mean_ret=fx, mean_ch=fx, diff=lambda v: "—" if nul(v) else "%+.0f%%" % (v * 100), auc=fx, p_adj=lambda v: "%.2g" % v, n_ch=n0f))
    h += note("<b>Association, not causation.</b> AUC 0.5 = no separation. Large samples make tiny differences significant; judge by effect size and sample size. A snapshot cannot show individual engagement trajectories; "
              "'usage change %' is a supplied change measure.", "warn")
    coh = empty("Needs subscription_start_date for cohort analysis")
    if S["subscription_start_date"].notna().any():
        c = L.assign(cohort=L["subscription_start_date"].dt.to_period("M").astype(str)).groupby("cohort")["churned"].agg(["sum", "count"]).reset_index(); c = c[c["count"] >= 30]
        if len(c):
            c["rate"] = c["sum"] / c["count"]; coh = sty(go.Figure(go.Bar(x=c["cohort"], y=c["rate"], marker_color=C["mi"], text=["n=%d" % n for n in c["count"]])), 340, "Churn rate by acquisition-month cohort (≥30 customers)"); coh.update_yaxes(tickformat=".0%")
    tr = empty("No engagement metric available")
    mm = next((m for m in ["logins_last_30_days", "usage_minutes"] if S[m].notna().any()), None)
    if mm and S["tenure_months"].notna().any():
        g = L.groupby(["tenure_stage", "churned"], observed=True)[mm].mean().reset_index(); fig = go.Figure()
        for v, nm, col in [(0, "Retained", C["te"]), (1, "Churned", C["co"])]:
            d = g[g["churned"] == v]; fig.add_trace(go.Scatter(x=d["tenure_stage"].astype(str), y=d[mm], name=nm, mode="lines+markers", line=dict(color=col, width=3)))
        tr = sty(fig, 340, "Average %s by tenure stage (cross-section, not individual trajectories)" % pretty(mm).lower())
    return pat_chart(first), h, coh, tr

# ------------------------------ D: segmentation ------------------------------
def seg_table():
    S = G["S"]; rows = []
    for s_ in ALLSEG:
        sub = S[S["segment"] == s_]; sa = sub[sub["active"] & sub["p"].notna()]; nl = int(sub["churned"].notna().sum()); lo, hi = wilson(sub["churned"].sum(), nl) if nl else (np.nan, np.nan)
        rows.append(dict(seg=s_, n=len(sub), act=len(sa), obs=sub["churned"].mean() if nl else np.nan, ci="—" if pd.isna(lo) else "%.0f%%–%.0f%%" % (lo * 100, hi * 100), p=sa["p"].mean(), fee=sa["monthly_fee"].mean(),
                         exp=sa["mrr_exp"].sum(min_count=1), act_=SEG2STRAT[s_], low="⚠ small n" if len(sub) < 30 else ""))
    return pd.DataFrame(rows)

def render_seg():
    if "S" not in G: return NA(2)
    S = G["S"]; d = seg_table(); c = G["segcfg"]
    h = note("Rules (configurable): <b>Onboarding</b> tenure ≤ %g mo and low logins or features; <b>Declining</b> usage change ≤ −%g%%; <b>Payment friction</b> ≥ %g failures; <b>Unresolved issues</b> ≥ %g; <b>Price-sensitive</b> used a discount; "
             "<b>High-value</b> fee ≥ %.0f%% quantile and High/Critical risk; <b>Low engagement</b> bottom-quartile logins and features; <b>Stable & loyal</b> no flags, ≥12 months, low risk. A customer can carry several flags; "
             "the <b>primary segment</b> is the first matching rule in this order: %s." % (c["onb"], c["decl"], c["pf"], c["unr"], c["hvq"] * 100, " → ".join(SEG_ORDER)), "info")
    h += tbl(d, [("seg", "Primary segment"), ("n", "Customers"), ("act", "Active & scored"), ("obs", "Observed churn"), ("ci", "95% CI"), ("p", "Avg predicted risk"), ("fee", "Avg fee"), ("exp", "Expected MRR exposure"), ("act_", "Suggested intervention"), ("low", "")],
             dict(n=n0f, act=n0f, obs=pc1, p=pc0, fee=money, exp=money))
    F = G["F"]; fl = pd.DataFrame([dict(flag=k, n=int(F[k].sum()), share=F[k].mean(), obs=S.loc[F[k], "churned"].mean() if S.loc[F[k], "churned"].notna().any() else np.nan) for k in SEG_ORDER])
    h += "<div class='sec'>Behavioural flags (a customer may have several)</div>" + tbl(fl, [("flag", "Flag"), ("n", "Customers"), ("share", "Share"), ("obs", "Observed churn")], dict(n=n0f, share=pc0, obs=pc1))
    h += note("%d customers carry two or more flags. Observed churn rates are associations; small segments have wide intervals." % int((F.sum(axis=1) >= 2).sum()), "warn")
    f = sty(go.Figure(go.Bar(x=d["seg"], y=d["act"], marker_color=C["te"])), 340, "Active customers by primary segment"); f.update_layout(xaxis=dict(tickangle=-25))
    return h, f

def seg_apply(onb, decl, pf, unr, hvq):
    if "S" not in G: return note("Run the analysis first.", "info"), empty("No data")
    G["segcfg"] = dict(onb=float(onb), decl=float(decl), pf=float(pf), unr=float(unr), hvq=float(hvq)); apply_segments(); return render_seg()

def render_cluster(k):
    if "S" not in G: return note("Run the analysis first.", "info"), empty("No data")
    S = G["S"]; nums = [f for f in G["info"]["nums"] if S[f].notna().mean() >= .7]
    if len(nums) < 2 or len(S) < 100: return note("Clustering needs at least 2 well-populated numeric features and 100 customers.", "warn"), empty("Not enough data")
    k = int(k); X = S[nums].fillna(S[nums].median()); D = [X] + [pd.get_dummies(S[c].astype(str), prefix=c).astype(float) for c in G["info"]["cats"]]
    Z = StandardScaler().fit_transform(pd.concat(D, axis=1)); lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z).labels_
    sil = silhouette_score(Z, lab, sample_size=min(2000, len(Z)), random_state=42); zs = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan); rows = []
    for c in range(k):
        m = lab == c; z = zs[m].mean().dropna(); top = z.reindex(z.abs().sort_values(ascending=False).index)[:3]
        desc = ", ".join("%s %s (%+.1fσ)" % ("high" if v > 0 else "low", pretty(f).lower(), v) for f, v in top.items()); sc = S[m]; nl = int(sc["churned"].notna().sum())
        rows.append(dict(cl="Cluster %d" % (c + 1), n=int(m.sum()), share=m.mean(), obs=sc["churned"].mean() if nl else np.nan, p=sc.loc[sc["active"], "p"].mean(), fee=sc["monthly_fee"].mean(), desc=desc))
    h = note("K-means (k=%d) on standardised numeric features (%s) plus one-hot contract/plan; missing values imputed with the median. Silhouette %.2f (closer to 1 = better separated; below 0.25 = weak structure). "
             "Descriptions were derived <b>after</b> inspecting each cluster's standardised averages; clusters are not objectively correct customer types." % (k, esc(", ".join(pretty(f) for f in nums)), sil), "info")
    h += tbl(pd.DataFrame(rows), [("cl", "Cluster"), ("n", "Customers"), ("share", "Share"), ("obs", "Observed churn"), ("p", "Avg predicted risk (active)"), ("fee", "Avg fee"), ("desc", "Most distinctive features vs overall")],
             dict(n=n0f, share=pc0, obs=pc1, p=pc0, fee=money))
    idx = np.random.default_rng(1).choice(len(Z), min(1500, len(Z)), replace=False); P = PCA(2, random_state=42).fit_transform(Z[idx]); fig = go.Figure()
    cols = [C["te"], C["co"], C["am"], C["mi"], "#8AA4FF", "#C9A0FF", "#FFD1A0", "#A0E7FF"]
    for c in range(k):
        m = lab[idx] == c; fig.add_trace(go.Scatter(x=P[m, 0], y=P[m, 1], mode="markers", name="Cluster %d" % (c + 1), marker=dict(size=6, opacity=.7, color=cols[c % len(cols)])))
    return h, sty(fig, 420, "Customer clusters (PCA projection of standardised features)")

# ------------------------------ E: customer profiles ------------------------------
def render_profile(cid):
    if "S" not in G or not cid: return note("Select a customer ID.", "info"), empty("No customer selected")
    S = G["S"]; M = G["M"]; r = S[S["customer_id"] == str(cid).strip()]
    if r.empty: return note("Customer ID not found.", "warn"), empty("Not found")
    r = r.iloc[0]
    h = "<div class='kg'>" + card("🎯", "Risk score", pc0(r["p"]), "calibrated probability (out-of-fold)" if M["ok"] else "model unavailable", RC.get(r["risk"], C["mu"])) + card("🏷️", "Risk category", r["risk"], "cut-offs %s" % G["cuts"], C["am"]) + \
        card("🧩", "Primary segment", esc(r["segment"]), "first matching rule", C["te"]) + card("💸", "Expected MRR exposure", money(r["mrr_exp"]), "fee %s × risk" % money(r["monthly_fee"]), C["co"]) + card("🛠️", "Suggested action", esc(r["strategy"]), "net est. %s" % money(r["net"]), C["mi"]) + "</div>"
    if not r["active"]: h += note("This customer is recorded as already churned; the score is an out-of-fold evaluation score, not a retention target.", "warn")
    rows = []
    for f in M["nums"]:
        v = r[f]; pr = (S[f] <= v).mean() * 100 if pd.notna(v) else np.nan
        rows.append(dict(f=pretty(f), v=fx(v, 1), t=fx(S[f].median(), 1), pct="—" if pd.isna(pr) else "%.0fth" % pr))
    for f in M["cats"]: rows.append(dict(f=pretty(f), v=esc(r[f]), t=esc(S[f].mode().iat[0]) if S[f].notna().any() else "—", pct="—"))
    h += "<div class='sec'>Customer values vs typical customer</div>" + tbl(pd.DataFrame(rows), [("f", "Feature"), ("v", "This customer"), ("t", "Typical (median / most common)"), ("pct", "Percentile")], dict(f=esc, v=str, t=str, pct=str))
    fig = empty("Model unavailable")
    if M["ok"]:
        Xr = prep_X(S.loc[[r.name]], M["nums"], M["cats"]); Cm, _ = contributions(M["final"], Xr, M["ref"]); cm = pd.Series(Cm[0], index=Xr.columns)
        top = cm.reindex(cm.abs().sort_values(ascending=False).index)[:8].iloc[::-1]
        fig = go.Figure(go.Bar(y=[pretty(x) for x in top.index], x=top.values * 100, orientation="h", marker_color=[C["co"] if v > 0 else C["te"] for v in top.values]))
        fig = sty(fig, 380, "What-if drivers: change in modeled risk (pts) vs replacing the value with a typical one"); fig.update_xaxes(title="percentage points")
        pos = cm[cm >= .01].sort_values(ascending=False)[:3]; neg = cm[cm <= -.01].sort_values()[:2]
        fmt = lambda f: "%s (%.0f)" % (pretty(f).lower(), r[f]) if f in M["nums"] and pd.notna(r[f]) else "%s (%s)" % (pretty(f).lower(), r[f])
        txt = "The model gives this customer a %s churn probability (%s). " % (pc0(r["p"]), r["risk"])
        txt += ("Compared with a typical customer, the factors raising modeled risk most are " + ", ".join(fmt(f) for f in pos.index) + ". ") if len(pos) else "No single factor stands out as raising the risk. "
        if len(neg): txt += "Factors lowering it: " + ", ".join(fmt(f) for f in neg.index) + ". "
        h += note(esc(txt) + "<br><b>Caution:</b> these are statistical associations, not verified reasons for leaving. A high complaint count may be linked to churn without having caused it.", "info")
    h += "<div class='sec'>Recommended retention action</div>" + rec_card(r)
    return h, fig

def predict_one(plan, contract, tenure, fee, pf, dsll, logins, chg, fu, tick, unres, csat):
    if "M" not in G or not G["M"]["ok"]: return note("Train a model first (run the analysis on labelled data).", "warn"), empty("No model")
    M = G["M"]; given = dict(subscription_plan=plan, contract_type=contract, tenure_months=tenure, monthly_fee=fee, payment_failures=pf, days_since_last_login=dsll, logins_last_30_days=logins,
                             usage_change_percentage=chg, features_used=fu, support_tickets=tick, unresolved_complaints=unres, customer_satisfaction=csat)
    row = {f: np.nan for f in M["feats"]}; used = []
    for k, v in given.items():
        if k in row and v is not None and v != "": row[k] = v; used.append(k)
    if not used: return note("Enter at least one value.", "warn"), empty("No input")
    X = prep_X(pd.DataFrame([row]), M["nums"], M["cats"]); p = float(M["final"].predict_proba(X)[0, 1]); cat = str(risk_of([p], G["cuts"])[0])
    Cm, _ = contributions(M["final"], X, M["ref"]); cm = pd.Series(Cm[0], index=X.columns); top = cm.reindex(cm.abs().sort_values(ascending=False).index)[:8].iloc[::-1]
    fig = sty(go.Figure(go.Bar(y=[pretty(x) for x in top.index], x=top.values * 100, orientation="h", marker_color=[C["co"] if v > 0 else C["te"] for v in top.values])), 340, "Drivers (pts vs typical value)")
    nm = [pretty(f) for f in M["feats"] if f not in used]
    h = "<div class='kg'>" + card("🎯", "Predicted churn probability", pc0(p), "calibrated model output", RC[cat]) + card("🏷️", "Risk category", cat, "cut-offs %s" % G["cuts"], C["am"]) + "</div>"
    h += note("Provided %d of %d predictors. Blank ones (%s) are imputed with training medians / most common values, which lowers reliability. This is a what-if estimate, not a verified outcome." % (len(used), len(M["feats"]), esc(", ".join(nm) if nm else "none")), "warn" if nm else "info")
    return h, fig

# ------------------------------ F: strategy studio ------------------------------
def render_strategy(cid):
    if "S" not in G or not cid: return note("Select a customer first.", "info"), ""
    r = G["S"][G["S"]["customer_id"] == str(cid).strip()]
    if r.empty: return note("Customer ID not found.", "warn"), ""
    r = r.iloc[0]; return rec_card(r) + note("Drafts use only supplied data and placeholders. Review and edit before use; the app never sends messages or changes accounts.", "info"), draft_msg(r)

def render_plan(n, seg):
    if "S" not in G: return pd.DataFrame()
    A = G["S"]; A = A[A["active"] & A["p"].notna()]
    if seg and seg != "All": A = A[A["segment"] == seg]
    rows = []
    for _, r in A.sort_values("prio_rank").head(int(n or 25)).iterrows():
        hyp, act, ch, tm = PLAY[r["segment"]]
        rows.append({"Customer ID": r["customer_id"], "Risk %": round(r["p"] * 100, 1), "Risk": r["risk"], "Segment": r["segment"], "Observed evidence": "; ".join(evidence(r)), "Recommended action": r["strategy"] + ": " + act,
                     "Channel": ch, "Timing": tm, "Cost ($)": round(r["cost"], 2), "Expected benefit ($, hypothetical)": None if pd.isna(r["benefit"]) else round(r["benefit"], 2), "Reviewer notes": ""})
    return pd.DataFrame(rows)
def export_plan(df):
    p = "churn_ai_retention_plan.csv"; pd.DataFrame(df).to_csv(p, index=False); return p
def export_msg(t):
    p = "churn_ai_message_draft.txt"; open(p, "w", encoding="utf-8").write(t or ""); return p

# ------------------------------ G: ROI simulator ------------------------------
def read_strats(df):
    try:
        d = pd.DataFrame(df).iloc[:, :3].copy(); d.columns = STRAT_COLS; base = DEFAULT_STRATS.set_index("Strategy"); d["Strategy"] = d["Strategy"].astype(str)
        if set(d["Strategy"]) != set(base.index): return DEFAULT_STRATS.copy()
        for c in STRAT_COLS[1:]: d[c] = pd.to_numeric(d[c], errors="coerce"); d[c] = d[c].fillna(d["Strategy"].map(base[c]))
        d["Cost per customer ($)"] = d["Cost per customer ($)"].clip(lower=0); d["Assumed effectiveness"] = d["Assumed effectiveness"].clip(0, 1); return d.reset_index(drop=True)
    except Exception: return DEFAULT_STRATS.copy()

def sim(thr, cap, budget, pos_only, scale):
    S = G["S"]; A = S[S["active"] & S["p"].notna() & S["value"].notna() & (S["p"] >= thr) & (S["strategy"] != "Monitor only")].copy()
    A["e"] = (A["eff"] * scale).clip(0, 1); A["ben"] = A["p"] * A["e"] * A["value"]; A["net_i"] = A["ben"] - A["cost"]; A["expo"] = A["p"] * A["value"]
    if pos_only: A = A[A["net_i"] > 0]
    A = A.sort_values("net_i", ascending=False)
    if cap and cap > 0: A = A.head(int(cap))
    if budget and budget > 0: A = A[A["cost"].cumsum() <= budget]
    return A

def render_roi(thr, cap, budget, pos_only, scale, strat):
    if "S" not in G: return note("Run the analysis first.", "info"), empty("No data"), empty("No data")
    if not G["M"]["ok"]: return note("The simulator needs a trained model: " + esc(G["M"]["why"]), "warn"), empty("No model"), empty("No model")
    G["strat"] = read_strats(strat); assign_strategy(); S = G["S"]; thr = float(thr if thr is not None else .3); scale = float(scale or 1); cfg = G["cfg"]
    A = sim(thr, cap, budget, pos_only, scale); e = econ(A["p"], A["value"], A["e"], A["cost"]); el = S[S["active"] & S["p"].notna() & S["value"].notna()]
    gross = float((el["p"] * el["value"]).sum()); nofee = int((S["active"] & S["monthly_fee"].isna()).sum())
    h = "<div class='kg'>" + card("🧮", "Gross contribution exposure", money(gross), "all %d active customers with a fee · Σ risk × value" % len(el), C["co"]) + \
        card("🎯", "Exposure in targeted group", money(e["exposure"]), "%d customers targeted" % len(A), C["am"]) + card("🛟", "Assumed incremental benefit", money(e["benefit"]), "Σ risk × effectiveness × value (hypothetical)", C["mi"]) + \
        card("💳", "Intervention cost", money(e["cost"]), "sum of strategy costs", C["te"]) + card("📈", "Net expected benefit", money(e["net"]), "benefit − cost", C["mi"] if e["net"] >= 0 else C["co"]) + \
        card("🔁", "Expected ROI", "—" if pd.isna(e["roi"]) else "%.0f%%" % (e["roi"] * 100), "net benefit ÷ cost", C["am"]) + "</div>"
    h += note("<b>Formulas.</b> Value = monthly fee × %.0f%% margin × %d months. Gross exposure = Σ pᵢ·Vᵢ. Benefit = Σ pᵢ·eᵢ·Vᵢ. Net = benefit − cost. ROI = net ÷ cost. Each customer receives one action (their primary segment's), so totals do not double count. "
              "<b>All effects are hypothetical</b>: observational churn data cannot show how many customers an action will save. Run A/B or holdout tests." % (cfg["margin"] * 100, cfg["horizon"]), "warn")
    if nofee: h += note("%d active customers have no monthly fee and are excluded from money calculations." % nofee, "warn")
    if len(A):
        g = A.groupby("strategy").agg(n=("p", "size"), expo=("expo", "sum"), ben=("ben", "sum"), cost=("cost", "sum")).reset_index(); g["net"] = g["ben"] - g["cost"]
        g["roi"] = np.where(g["cost"] > 0, g["net"] / g["cost"], np.nan); g["be"] = np.where(g["expo"] > 0, g["cost"] / g["expo"], np.nan)
        h += "<div class='sec'>Recommended mix: each customer gets one action (additive)</div>" + tbl(g.sort_values("net", ascending=False), [("strategy", "Strategy"), ("n", "Customers"), ("expo", "Exposure"), ("ben", "Assumed benefit"), ("cost", "Cost"), ("net", "Net"), ("roi", "ROI"), ("be", "Break-even effectiveness")],
                                                                                                              dict(n=n0f, expo=money, ben=money, cost=money, net=mcol, roi=lambda v: "—" if nul(v) else "%.0f%%" % (v * 100), be=lambda v: "—" if nul(v) else "%.1f%%" % (v * 100)))
    else: h += note("No customers match the current threshold / capacity / budget settings.", "warn")
    sd = G["strat"].set_index("Strategy"); F = G["F"]; rows = []
    for s_ in SEG_ORDER[:-1]:
        st = SEG2STRAT[s_]; sub = S[F[s_] & S["active"] & S["p"].notna() & S["value"].notna() & (S["p"] >= thr)]
        ee = econ(sub["p"], sub["value"], sd.loc[st, "Assumed effectiveness"] * scale, sd.loc[st, "Cost per customer ($)"])
        rows.append(dict(st=st, n=len(sub), expo=ee["exposure"], ben=ee["benefit"], cost=ee["cost"], net=ee["net"], roi=ee["roi"]))
    h += "<div class='sec'>Strategy comparison if each is applied to ALL matching flagged customers (overlapping: do not add rows)</div>" + tbl(pd.DataFrame(rows).sort_values("net", ascending=False), [("st", "Strategy"), ("n", "Customers"), ("expo", "Exposure"), ("ben", "Assumed benefit"), ("cost", "Cost"), ("net", "Net"), ("roi", "ROI")],
                                                                                                                                                  dict(n=n0f, expo=money, ben=money, cost=money, net=mcol, roi=lambda v: "—" if nul(v) else "%.0f%%" % (v * 100)))
    if len(A) and e["benefit"] > 0:
        p0 = float(A["p"].mean()); rel = float((A["p"] * A["e"]).sum() / A["p"].sum()); p1 = p0 * (1 - rel)
        if 0 < p0 < 1 and p0 > p1:
            n = (stats.norm.ppf(.975) + stats.norm.ppf(.8)) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p0 - p1) ** 2
            h += note("<b>How to measure whether it works:</b> randomise targeted customers into treatment and holdout groups. With a baseline risk of about %.0f%% and the assumed relative reduction of %.0f%%, detecting the effect at 80%% power and 5%% significance needs roughly <b>%s customers per arm</b>." % (p0 * 100, rel * 100, format(int(np.ceil(n)), ",")), "info")
    f1 = sty(go.Figure(go.Bar(x=["Gross exposure (all active)", "Exposure (targeted)", "Assumed benefit", "Cost", "Net benefit"], y=[gross, e["exposure"], e["benefit"], e["cost"], e["net"]], marker_color=[C["co"], C["am"], C["mi"], C["te"], C["mi"] if e["net"] >= 0 else C["co"]])), 360, "Exposure → benefit → net (hypothetical)")
    ms = [.25, .5, .75, 1, 1.25, 1.5, 2]; nets = []; rois = []
    for m in ms:
        Am = sim(thr, cap, budget, pos_only, scale * m); em = econ(Am["p"], Am["value"], Am["e"], Am["cost"]); nets.append(em["net"]); rois.append(em["roi"])
    f2 = go.Figure(go.Bar(x=["%gx" % m for m in ms], y=nets, marker_color=[C["mi"] if v >= 0 else C["co"] for v in nets], text=["ROI —" if pd.isna(x) else "ROI %.0f%%" % (x * 100) for x in rois]))
    f2 = sty(f2, 360, "Sensitivity: net benefit if true effectiveness is a multiple of the assumption"); f2.update_yaxes(title="Net benefit ($)")
    return h, f1, f2

# ------------------------------ H: model page ------------------------------
def render_model():
    if "M" not in G: return (note("Run the analysis first.", "info"),) + NA(8)
    M = G["M"]
    if not M["ok"]:
        e = empty("Model unavailable"); return (note("Model training/evaluation unavailable: " + esc(M["why"]), "warn"), e, e, e, e, e, e, e, "")
    R = M["R"]; yte = M["yte"]; pc = M["cal_p"]; thr = M["thr"]; best = M["best"]; t = M["test_cal"]; cfg = G["cfg"]; rows = []
    for n, r in R.items():
        x = r["test"]; rows.append(dict(m=n + (" ★" if n == best else ""), k="raw score (class-weighted)", cv=r["cv_auc"], roc=x["roc"], pr=x["pr"], prec=x["prec"], rec=x["rec"], f1=x["f1"], br=x["brier"], thr=r["thr"]))
    rows.append(dict(m=best + " + sigmoid calibration (SELECTED)", k="calibrated probability", cv=M["cv_cal"], roc=t["roc"], pr=t["pr"], prec=t["prec"], rec=t["rec"], f1=t["f1"], br=t["brier"], thr=thr))
    flagged = int((pc >= thr).sum())
    h = note("Held-out test set: %d customers (%.1f%% churned). Preprocessing and models were fitted on the %d-customer training split only; the model was chosen by training cross-validation ROC-AUC and the threshold by cross-validated training predictions "
             "(objective: %s), never by the test set. Test ROC-AUC 95%% bootstrap interval: %s–%s." % (M["n_test"], yte.mean() * 100, M["n_train"], esc(cfg["obj"]), fx(M["auc_ci"][0], 3), fx(M["auc_ci"][1], 3)), "info")
    h += tbl(pd.DataFrame(rows), [("m", "Model"), ("k", "Score type"), ("cv", "Train CV ROC-AUC"), ("roc", "Test ROC-AUC"), ("pr", "Test PR-AUC"), ("prec", "Precision"), ("rec", "Recall"), ("f1", "F1"), ("br", "Brier"), ("thr", "Threshold")],
             dict(cv=lambda v: fx(v, 3), roc=lambda v: fx(v, 3), pr=lambda v: fx(v, 3), prec=lambda v: fx(v, 3), rec=lambda v: fx(v, 3), f1=lambda v: fx(v, 3), br=lambda v: fx(v, 3), thr=fx))
    h += note("<b>Trade-off at threshold %.2f:</b> the model flags %d of %d test customers, catching %d of %d churners (recall %.0f%%). Precision is %.0f%%, so about %.0f of every 100 flagged customers are false alarms (%d contacted unnecessarily). "
              "Lowering the threshold catches more churners but raises false alarms and contact cost; raising it does the opposite. Raw class-weighted scores are not probabilities; the calibrated score is (Brier %.3f vs %.3f raw)." %
              (thr, flagged, len(yte), t["tp"], t["tp"] + t["fn"], t["rec"] * 100, t["prec"] * 100, (1 - t["prec"]) * 100, t["fp"], t["brier"], R[best]["test"]["brier"]), "info")
    h += note("Metrics come from a held-out sample of " + ("<b>synthetic data</b> and say nothing about real-world accuracy." if cfg["demo"] else "your data; they are estimates and can change with new data.") + " SHAP is not used: local explanations use transparent what-if replacement (tab E), global importance uses permutation importance.", "warn")
    f1 = go.Figure()
    for n, r in R.items(): a, b, _ = roc_curve(yte, r["p"]); f1.add_trace(go.Scatter(x=a, y=b, name="%s (%.2f)" % (n, r["test"]["roc"]), mode="lines"))
    a, b, _ = roc_curve(yte, pc); f1.add_trace(go.Scatter(x=a, y=b, name="Selected calibrated (%.2f)" % t["roc"], line=dict(width=4, color=C["mi"])))
    f1.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(dash="dash", color=C["mu"]))); f1 = sty(f1, 380, "ROC curves (held-out)"); f1.update_xaxes(title="False positive rate"); f1.update_yaxes(title="True positive rate")
    f2 = go.Figure()
    for n, r in R.items(): a, b, _ = precision_recall_curve(yte, r["p"]); f2.add_trace(go.Scatter(x=b, y=a, name="%s (%.2f)" % (n, r["test"]["pr"]), mode="lines"))
    a, b, _ = precision_recall_curve(yte, pc); f2.add_trace(go.Scatter(x=b, y=a, name="Selected calibrated", line=dict(width=4, color=C["mi"])))
    f2.add_hline(y=float(yte.mean()), line_dash="dash", line_color=C["mu"], annotation_text="base rate"); f2 = sty(f2, 380, "Precision-recall curves (held-out)"); f2.update_xaxes(title="Recall"); f2.update_yaxes(title="Precision")
    f3 = go.Figure(go.Heatmap(z=[[t["tn"], t["fp"]], [t["fn"], t["tp"]]], x=["Predicted retained", "Predicted churn"], y=["Actually retained", "Actually churned"], colorscale=[[0, C["sf"]], [1, C["te"]]], text=[[t["tn"], t["fp"]], [t["fn"], t["tp"]]], texttemplate="%{text}", showscale=False))
    f3 = sty(f3, 340, "Confusion matrix at threshold %.2f (held-out)" % thr); f3.update_yaxes(autorange="reversed")
    f4 = go.Figure([go.Histogram(x=pc[yte == 0], name="Retained", xbins=dict(start=0, end=1, size=.05), marker_color=C["te"], opacity=.7), go.Histogram(x=pc[yte == 1], name="Churned", xbins=dict(start=0, end=1, size=.05), marker_color=C["co"], opacity=.7)])
    f4.update_layout(barmode="overlay"); f4.add_vline(x=thr, line_dash="dash", line_color=C["am"]); f4 = sty(f4, 340, "Prediction distribution by actual outcome (held-out)"); f4.update_xaxes(title="Calibrated churn probability")
    f5 = go.Figure()
    for nm, p_, col in [("Raw " + best, R[best]["p"], C["co"]), ("Calibrated", pc, C["mi"])]:
        try: fr, mp = calibration_curve(yte, p_, n_bins=8, strategy="quantile"); f5.add_trace(go.Scatter(x=mp, y=fr, name=nm, mode="lines+markers", line=dict(color=col)))
        except Exception: pass
    f5.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect", line=dict(dash="dash", color=C["mu"]))); f5 = sty(f5, 340, "Calibration diagram (held-out)"); f5.update_xaxes(title="Mean predicted probability"); f5.update_yaxes(title="Observed churn rate")
    g = np.linspace(.05, .9, 35); P = pc[:, None] >= g[None, :]; tp = (P & (yte[:, None] == 1)).sum(0); fp = (P & (yte[:, None] == 0)).sum(0)
    prc = tp / np.maximum(tp + fp, 1); rcl = tp / max(int(yte.sum()), 1); f1s = 2 * prc * rcl / np.maximum(prc + rcl, 1e-9)
    f6 = go.Figure([go.Scatter(x=g, y=prc, name="Precision"), go.Scatter(x=g, y=rcl, name="Recall"), go.Scatter(x=g, y=f1s, name="F1")]); f6.add_vline(x=thr, line_dash="dash", line_color=C["am"], annotation_text="chosen")
    f6 = sty(f6, 340, "Threshold trade-offs (held-out)"); f6.update_xaxes(title="Decision threshold")
    imp = M["imp"]
    if imp is not None and len(imp):
        d = imp.head(12).iloc[::-1]; f7 = go.Figure(go.Bar(y=[pretty(x) for x in d["feature"]], x=d["imp"], orientation="h", error_x=dict(type="data", array=d["sd"].values), marker_color=C["te"]))
        f7 = sty(f7, 380, "Permutation importance on held-out data (%s): drop in ROC-AUC" % best)
    else: f7 = empty("Importance unavailable")
    grp = np.where((pc >= thr) & (yte == 1), "TP caught churner", np.where((pc >= thr) & (yte == 0), "FP false alarm", np.where((pc < thr) & (yte == 1), "FN missed churner", "TN correctly retained")))
    topf = [f for f in (imp["feature"].tolist() if imp is not None else []) if f in M["nums"]][:6] or M["nums"][:6]
    ea = M["Xte"][topf].assign(group=grp).groupby("group")[topf].mean().reset_index(); ea.insert(1, "n", [int((grp == g_).sum()) for g_ in ea["group"]])
    tb = "<div class='sec'>Error analysis: average feature values by outcome group</div>" + tbl(ea, [("group", "Group"), ("n", "n")] + [(f, pretty(f)) for f in topf], {f: (lambda v: fx(v, 1)) for f in topf})
    S = G["S"]; A = S[S["active"] & S["p"].notna()]; ov = A["risk"].isin(["High", "Critical"]).mean(); Xte = M["Xte"]
    for c in M["cats"]:
        sr = []
        for lv in sorted(Xte[c].unique()):
            m = (Xte[c] == lv).values; n = int(m.sum()); yy = yte[m]; pp = pc[m]; ok = n >= 40 and yy.sum() >= 8 and (n - yy.sum()) >= 8
            sa = A[A[c].astype(str) == lv]; fr = sa["risk"].isin(["High", "Critical"]).mean() if len(sa) else np.nan
            rv = "review" if pd.notna(fr) and ov > 0 and len(sa) >= 30 and (fr / ov > 1.5 or fr / ov < .67) else ""
            sr.append(dict(lv=lv, n=n, ch=yy.mean() if n else np.nan, auc=roc_auc_score(yy, pp) if ok else np.nan, rec=((pp >= thr) & (yy == 1)).sum() / yy.sum() if yy.sum() >= 8 else np.nan, fr=fr, rv=rv))
        tb += "<div class='sec'>Subgroup performance and targeting rate by %s</div>" % pretty(c).lower() + tbl(pd.DataFrame(sr), [("lv", "Group"), ("n", "Test n"), ("ch", "Test churn rate"), ("auc", "Test ROC-AUC"), ("rec", "Recall"), ("fr", "Share flagged High/Critical (active)"), ("rv", "Human review?")],
                                                                                                                 dict(n=n0f, ch=pc1, auc=fx, rec=pc0, fr=pc1))
    tb += note("<b>Fairness.</b> No demographic attribute is used as a feature" + (" (sensitive-looking columns excluded: " + esc(", ".join(G["info"]["sens"])) + ")" if G["info"]["sens"] else "") + ". This table checks performance and targeting-rate gaps across plan/contract only; it is <b>not</b> a full fairness audit. "
               "Groups flagged 'review' have a flag rate 1.5× above or below average and need human review before offers differ between groups.", "warn")
    tb += note("<b>Monitoring status.</b> Implemented: input-drift PSI for batch uploads (tab B). Not implemented: outcome and performance drift tracking (needs future labelled snapshots).", "info")
    tb += "".join(note(esc(x), "warn") for x in M["warn"])
    return h, f1, f2, f3, f4, f5, f6, f7, tb

# ------------------------------ I: data quality, self-tests, methodology ------------------------------
def role_of(k):
    if k == "customer_id": return "Identifier (not a predictor)"
    if k == "churned": return "Churn target"
    if k in POST_F: return "Post-churn information (never used)"
    if k in FIN_F: return "Financial field (value only, not a predictor)"
    if k in DATE_F: return "Date (derives tenure / renewal timing / cohorts)"
    if k in NUM_F: return "Predictor (numeric)" + (" + financial value" if k == "monthly_fee" else "")
    if k in CAT_F: return "Predictor (categorical)"
    return "Other"

def render_dq():
    if "S" not in G: return note("Run the analysis first.", "info"), ""
    df = G["df"]; info = G["info"]; ex = dict(info["excl"])
    h = note("Customers analysed: <b>%d</b> (file rows %d). " % (len(df), G["raw_n"]) + ("<b>ILLUSTRATIVE SYNTHETIC DATA</b>: not real customers." if G["demo"] else "Uploaded data. It is never written to the project ZIP."), "ok")
    h += "".join(note(esc(w), "warn") for w in G["W"]) or note("No data-quality issues detected.", "ok")
    rows = []
    for k in SYN:
        st = "not provided" if k not in info["avail"] else ("used as predictor" if k in info["nums"] + info["cats"] else ("excluded: " + ex[k] if k in ex else "used for its role"))
        rows.append(dict(k=k, role=role_of(k), st=st, cov=df[k].notna().mean() if k in info["avail"] else 0.0))
    h += "<div class='sec'>Field roles: predictors, target, identifiers, post-churn and financial fields</div>" + tbl(pd.DataFrame(rows), [("k", "Field"), ("role", "Role"), ("st", "Status"), ("cov", "Row coverage")], dict(cov=pc0))
    h += note("Event-level history is not ingested directly: supply pre-aggregated window features (e.g. logins_last_30_days, usage_change_percentage) computed over a window that ends <b>before</b> the churn outcome period.", "info")
    th = "<div class='sec'>Self-test results (executed when this app started)</div>" + "".join(note(("✅ " if ok else "❌ ") + esc(n) + " — " + esc(d), "ok" if ok else "err") for n, ok, d in TESTS)
    return h, th

def run_selftests(verbose=True):
    T = []
    def chk(name, fn):
        try: ok, det = fn(); T.append((name, bool(ok), det))
        except Exception as e: T.append((name, False, type(e).__name__ + ": " + str(e)[:80]))
    def t_wil(): lo, hi = wilson(20, 100); return lo < .2 < hi and 0 <= lo and hi <= 1, "CI for 20/100 = %.3f-%.3f" % (lo, hi)
    def t_econ(): e = econ([.5, .5], [100, 100], [.2, .2], [10, 10]); return abs(e["exposure"] - 100) < 1e-9 and abs(e["benefit"] - 20) < 1e-9 and abs(e["net"]) < 1e-9 and abs(e["roi"]) < 1e-9, "exposure %.0f, benefit %.0f, net %.0f, ROI %.2f" % (e["exposure"], e["benefit"], e["net"], e["roi"])
    def t_thr():
        y = np.array([0] * 50 + [1] * 50); p = np.r_[np.linspace(.05, .4, 50), np.linspace(.6, .95, 50)]; t = pick_threshold(y, p, OBJS[0], 100, 10); return .35 <= t <= .6, "F1-optimal threshold %.2f (not a fixed 0.5)" % t
    def t_psi(): x = np.random.default_rng(0).normal(size=2000); a = psi(x, x); b = psi(x, x + 1.5); return a < .01 and b > .25, "PSI same %.4f, shifted %.2f" % (a, b)
    def t_val(): v = value_of(pd.Series([10.0, np.nan]), .5, 12); return v.iloc[0] == 60 and pd.isna(v.iloc[1]), "missing fee stays missing (never zero)"
    def t_leak():
        d = pd.DataFrame(dict(customer_id=["a", "b", "c", "d"], tenure=[1, 5, 9, 12], monthly_fee=[10, 20, 30, 15], churned=["yes", "no", "no", "no"], cancellation_date=["2026-01-01", None, None, None]))
        _, _, info = clean(d); f = info["nums"] + info["cats"]; return "cancellation_date" not in f and "churned" not in f and "tenure_months" in f, "features = %s" % f
    def t_tgt(): y = parse_target(pd.Series(["Yes", "no", "1", "0", "Churned", "Active", "?"])); return list(y.fillna(-1)) == [1, 0, 1, 0, 1, 0, -1], "yes/no/1/0/churned/active parsed; unknown stays missing"
    def t_model():
        d, _, info = clean(make_synth(900, seed=3).drop(columns=["cancellation_date"])); y = d["churned"].astype(int).values; X = prep_X(d, info["nums"], info["cats"])
        tr, te = train_test_split(np.arange(len(d)), test_size=.3, stratify=y, random_state=0); m = zoo_models(info["nums"], info["cats"])["Logistic Regression"].fit(X.iloc[tr], y[tr])
        a = roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]); return a > .65, "held-out ROC-AUC %.3f on a fresh synthetic sample" % a
    for n, f in [("Wilson interval", t_wil), ("ROI formulas (exposure, benefit, net, ROI)", t_econ), ("Threshold chosen by objective", t_thr), ("PSI drift metric", t_psi), ("Missing revenue is never zero", t_val),
                 ("Leakage guard (cancellation date / target excluded)", t_leak), ("Target parsing", t_tgt), ("Pipeline beats chance on held-out data", t_model)]: chk(n, f)
    if verbose:
        for n, ok, d in T: print(("PASS " if ok else "FAIL ") + n + " :: " + d)
    return T

# ------------------------------ J: AI agent (tool-using) ------------------------------
def tool_summary():
    S = G["S"]; M = G["M"]; A = S[S["active"]]; sc = A[A["p"].notna()]; hr = sc[sc["risk"].isin(["High", "Critical"])]
    s = "Customers: %d; active: %d; observed churn rate: %s. " % (len(S), len(A), pc1(S["churned"].mean()))
    if M["ok"]: s += "High+Critical active customers: %d (%.1f%%); expected MRR at risk: %s; held-out ROC-AUC %.3f, PR-AUC %.3f." % (len(hr), 100 * len(hr) / max(len(sc), 1), money(sc["mrr_exp"].sum(min_count=1)), M["test_cal"]["roc"], M["test_cal"]["pr"])
    else: s += "Model unavailable: " + M["why"]
    return s
def tool_exposure():
    S = G["S"]; cfg = G["cfg"]; sc = S[S["active"] & S["p"].notna()]
    return "Expected MRR at risk %s; contribution exposure %s (margin %.0f%%, horizon %d months); gross exposure, not recoverable value. Customers without a fee excluded: %d." % (money(sc["mrr_exp"].sum(min_count=1)), money(sc["exp_contrib"].sum(min_count=1)), cfg["margin"] * 100, cfg["horizon"], int(sc["monthly_fee"].isna().sum()))
def tool_segments():
    d = seg_table(); d = d[d["n"] > 0].sort_values("p", ascending=False)
    return "Segments (customers, observed churn, avg predicted risk): " + "; ".join("%s: %d, %s, %s" % (r.seg, r.n, pc1(r.obs), pc0(r.p)) for r in d.itertuples())
def tool_lookup(cid):
    r = G["S"][G["S"]["customer_id"] == cid]
    if r.empty: return "Customer %s not found." % cid
    r = r.iloc[0]; return "%s: risk %s (%s); segment %s; monthly fee %s; expected MRR exposure %s; active: %s." % (cid, pc0(r["p"]), r["risk"], r["segment"], money(r["monthly_fee"]), money(r["mrr_exp"]), bool(r["active"]))
def tool_explain(cid):
    r = G["S"][G["S"]["customer_id"] == cid]
    return "" if r.empty else "Model drivers for %s (association, not cause): %s." % (cid, r.iloc[0]["drivers"] or "n/a")
def tool_recommend(cid):
    r = G["S"][G["S"]["customer_id"] == cid]
    if r.empty: return ""
    r = r.iloc[0]; return "Recommendation for %s: %s. Evidence: %s. Cost %s; hypothetical benefit %s. %s" % (cid, r["strategy"], "; ".join(evidence(r)), money(r["cost"]), money(r["benefit"]), PLAY[r["segment"]][1])
def tool_top(n=5):
    t = G["S"][G["S"]["active"] & G["S"]["p"].notna()].sort_values("prio_rank").head(n)
    return "Top priorities (by expected net benefit): " + "; ".join("%s (risk %s, %s, %s)" % (r.customer_id, pc0(r.p), r.segment, r.strategy) for r in t.itertuples())
def tool_simulate():
    A = sim(.3, 300, 0, True, 1.0); e = econ(A["p"], A["value"], A["e"], A["cost"]); roi = "—" if pd.isna(e["roi"]) else "%.0f%%" % (e["roi"] * 100)
    return "Simulation (risk ≥ 30%%, up to 300 contacts, positive-net customers only, assumed effectiveness): %d customers; cost %s; assumed benefit %s; net %s; ROI %s. Hypothetical: validate with an A/B test." % (len(A), money(e["cost"]), money(e["benefit"]), money(e["net"]), roi)

def check_numbers(ans, ctx):
    cn = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", ctx.replace(",", ""))]; bad = []
    for s in re.findall(r"\d+(?:\.\d+)?", ans.replace(",", "")):
        v = float(s)
        if v <= 10 or 1990 <= v <= 2100: continue
        if not any(abs(v - c) <= max(.06, .015 * abs(c)) for c in cn): bad.append(s)
    return sorted(set(bad))

def agent_answer(q, key, model):
    if "S" not in G: return "Run the analysis first."
    q = (q or "").strip() or "Give me an executive summary and the top retention priorities."; ql = q.lower(); ids = set(G["S"]["customer_id"])
    cid = next((t for t in re.findall(r"[A-Za-z0-9\-_]+", q) if t in ids), None); parts = []
    if cid: parts += [("customer_lookup", tool_lookup(cid)), ("prediction_explanation", tool_explain(cid)), ("retention_recommendation", tool_recommend(cid))]
    if any(w in ql for w in ["revenue", "exposure", "mrr", "money", "at risk"]): parts.append(("revenue_exposure", tool_exposure()))
    if any(w in ql for w in ["segment", "vulnerable", "group"]): parts.append(("segment_analysis", tool_segments()))
    if any(w in ql for w in ["simulat", "roi", "strategy", "worth", "prioriti"]): parts.append(("retention_simulation", tool_simulate()))
    if any(w in ql for w in ["which customers", "most likely", "top", "prioriti", "who"]): parts.append(("customer_ranking", tool_top()))
    if "report" in ql: parts.append(("report", "Use the 'Generate management report' button in tab I to download the full report."))
    if not parts: parts = [("dataset_summary", tool_summary()), ("revenue_exposure", tool_exposure()), ("customer_ranking", tool_top())]
    parts = [(n, t) for n, t in parts if t]; ctx = "\n".join("[%s] %s" % (n, t) for n, t in parts)
    base = "### Answer (deterministic, computed by tools)\n" + "\n".join("- **%s**: %s" % (n, t) for n, t in parts) + "\n\n*Tools used: " + ", ".join(n for n, _ in parts) + ". All numbers come from the current dataset.*"
    k = (key or "").strip() or os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        try:
            from google.colab import userdata; k = userdata.get("ANTHROPIC_API_KEY")
        except Exception: k = None
    if not k: return base + "\n\n*No API key set: rule-based agent used. Add a key (session only) for free-form language answers.*"
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", headers={"x-api-key": k, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": (model or "claude-sonnet-5-5").strip(), "max_tokens": 900, "system": "You are a retention analytics assistant. Answer ONLY from the TOOL RESULTS. Never invent numbers, customers or confidence scores; never override model outputs. Label hypotheses as hypotheses. Be concise.",
                                "messages": [{"role": "user", "content": "TOOL RESULTS:\n" + ctx + "\n\nQUESTION: " + q}]}, timeout=60)
        if r.status_code != 200: return base + "\n\n*LLM call failed (HTTP %d); showing deterministic answer.*" % r.status_code
        ans = "".join(b.get("text", "") for b in r.json().get("content", [])); bad = check_numbers(ans, ctx)
        return "### LLM-written answer (grounded in tool results; hypotheses are labelled)\n" + ans + (("\n\n⚠️ Numbers not found in tool results: " + ", ".join(bad) + ". Treat as unverified.") if bad else "\n\n✅ Numeric check: all figures match tool results.") + "\n\n---\n" + base
    except Exception as e: return base + "\n\n*LLM call failed (%s); showing deterministic answer.*" % type(e).__name__

# ------------------------------ report & project export ------------------------------
METH = """### Methodology
- **Target and period:** the churn column as supplied. Synthetic demo label = cancelled within 90 days after a 2026-06-30 snapshot. Cancellation dates, post-churn behaviour, historical revenue and lifetime value are never predictors.
- **Pipeline:** stratified 75/25 split *before* any fitting. Median imputation + scaling (logistic), median imputation (trees), constant imputation + one-hot for categories. Class imbalance via class weights and threshold choice; no resampling of the whole dataset.
- **Models:** Logistic Regression, Random Forest, Gradient Boosting. The best by 5-fold cross-validated ROC-AUC on training data is wrapped in sigmoid calibration (fitted on training data only).
- **Threshold:** chosen on cross-validated training predictions by a configurable objective (max F1, max F2, or cost of missed churner vs false alarm). Never fixed at 0.5 and never tuned on the test set.
- **Customer scores:** out-of-fold calibrated probabilities (no customer is scored by a model that trained on them). New/batch customers use the final model fitted on all labelled data.
- **Risk categories:** Moderate ≥ observed base rate, High ≥ chosen threshold, Critical ≥ max(60%, High + 15 pts) when 'auto'; otherwise your sliders.
- **Explainability:** local = what-if replacement of each value by the typical value; global = permutation importance on held-out data. Association only; SHAP is not used.
- **Priority:** expected net benefit = risk × assumed effectiveness × value − cost, where value = monthly fee × margin × horizon. Missing fees are never zero.
- **ROI:** exposure = Σ p·V; benefit = Σ p·e·V; net = benefit − cost; ROI = net ÷ cost. Effectiveness values are hypothetical; measure true impact with randomised holdouts.
- **Limitations:** observational data cannot prove causes or intervention effects; synthetic metrics do not reflect real-world accuracy; fairness checks are partial; outcome monitoring is not implemented.
- **Privacy:** pseudonymous IDs only; uploads are never saved into the project ZIP; API keys stay in the session; only aggregated tool results (and one customer's computed facts when you name an ID) go to an LLM.
"""
REQS = "gradio>=4.44,<6\nplotly\nscipy\nscikit-learn>=1.2\nrequests\npandas\nnumpy\n"
README = "# CHURN / AI\nPredict risk. Understand behavior. Retain intelligently.\n\nInteractive customer-retention intelligence: churn prediction (logistic regression, random forest, gradient boosting with calibration), behavioural patterns, segmentation, individual explanations, retention strategy studio, ROI simulator and a tool-using agent.\n\n## Run\nGoogle Colab: paste `app.py` into one cell and run it.\nLocally: `pip install -r requirements.txt && python app.py`\n\n## Data\n`data/sample_subscription_data.csv` is **fictional synthetic data**. Upload your own CSV in the app (pseudonymous IDs recommended). Never commit real customer data or API keys.\n\n## Notes\nPredictions show association, not causation. Simulator effects are hypothetical until validated with A/B tests. Optional LLM key: set `ANTHROPIC_API_KEY` as an environment variable or Colab secret.\n\n## Stack\nPython, Gradio, Plotly, scikit-learn, SciPy, pandas\n"
GITHUB = """### Export & publish to GitHub
1. Download the project ZIP below and unzip it (app.py, README, requirements, docs, fictional sample data).
2. Create an empty GitHub repository (no README / .gitignore), then upload the unzipped files (Add file → Upload files).
3. Never upload real customer data, generated risk lists, or API keys. Keep keys in Colab Secrets (`ANTHROPIC_API_KEY`).
"""

def make_report():
    if "S" not in G: return None, note("Run the analysis first.", "info")
    S = G["S"]; M = G["M"]; cfg = G["cfg"]; info = G["info"]; A = S[S["active"] & S["p"].notna()]; B = lambda t: "<h2>" + t + "</h2>"; parts = []
    parts.append("<h1>CHURN / AI — Management Report</h1><p class='m'>Predict risk. Understand behavior. Retain intelligently.</p>" + ("<p class='w'>Illustrative synthetic data: fictional customers; not real results.</p>" if cfg["demo"] else ""))
    parts.append(B("Executive summary") + "<p>%s</p><p>%s</p>" % (esc(tool_summary()), esc(tool_exposure())))
    parts.append(B("Business problem") + "<p>A subscription business faces rising churn despite marketing spend. This analysis identifies customers most likely to churn, behaviours associated with churn, vulnerable segments, and the retention actions with the best expected net benefit.</p>")
    parts.append(B("Data coverage and quality") + "<p>%d customers; predictors used: %s.</p><ul>" % (len(S), esc(", ".join(pretty(f) for f in info["nums"] + info["cats"]))) + "".join("<li>%s</li>" % esc(w) for w in G["W"]) + "</ul>")
    if M["ok"]:
        t = M["test_cal"]; parts.append(B("Modeling methodology and evaluation") + "<p>Selected model: %s with sigmoid calibration. Held-out test (n=%d): ROC-AUC %.3f (95%% CI %.3f–%.3f), PR-AUC %.3f, precision %.3f, recall %.3f, F1 %.3f at threshold %.2f (%s). Metrics do not guarantee future accuracy%s</p>" %
                      (esc(M["best"]), M["n_test"], t["roc"], M["auc_ci"][0], M["auc_ci"][1], t["pr"], t["prec"], t["rec"], t["f1"], M["thr"], esc(cfg["obj"]), "; they come from synthetic data." if cfg["demo"] else "."))
    else: parts.append(B("Modeling") + "<p>Model unavailable: %s</p>" % esc(M["why"]))
    d = seg_table(); d = d[d["n"] > 0]
    parts.append(B("Behavioural patterns and segments") + tbl(d, [("seg", "Segment"), ("n", "Customers"), ("obs", "Observed churn"), ("p", "Avg predicted risk"), ("exp", "Expected MRR exposure"), ("act_", "Suggested intervention")], dict(n=n0f, obs=pc1, p=pc0, exp=money)))
    if M["ok"]:
        top = A.sort_values("prio_rank").head(10); parts.append(B("Customer risk examples (pseudonymous IDs)") + tbl(top, [("customer_id", "Customer"), ("p", "Risk"), ("risk", "Category"), ("segment", "Segment"), ("strategy", "Suggested action"), ("net", "Est. net benefit")], dict(p=pc0, net=mcol)))
        parts.append(B("Retention priorities and scenario (hypothetical)") + "<p>%s</p>" % esc(tool_simulate()))
        f = go.Figure(go.Bar(x=RISKS, y=[(A["risk"] == c).sum() for c in RISKS], marker_color=[RC[c] for c in RISKS])); f.update_layout(title="Active customers by risk category", height=320)
        parts.append(f.to_html(full_html=False, include_plotlyjs="cdn"))
    parts.append(B("How to measure whether retention works") + "<p>Randomly assign eligible high-risk customers to treatment and holdout groups, hold contact capacity constant, and compare churn over the same period. Only such experiments establish incremental impact.</p>")
    parts.append(B("Limitations and ethics") + "<ul><li>Associations are not causes; explanations are not verified reasons for leaving.</li><li>No messages are sent automatically; humans review every action.</li><li>Do not use sensitive attributes to target or exclude customers; review flagged groups.</li><li>Monitoring beyond input drift is not implemented.</li></ul>")
    css = "body{font-family:Inter,Arial,sans-serif;background:#101722;color:#F3F7FA;max-width:1000px;margin:30px auto;padding:0 18px}h1{color:#42C9B4}h2{color:#7BE1C8;border-left:4px solid #42C9B4;padding-left:10px;margin-top:30px}.m{color:#9DADBC}.w{background:#F4BC7722;border-left:4px solid #F4BC77;padding:8px 12px}table{border-collapse:collapse;width:100%;background:#18212D}th{background:#22303f;color:#7BE1C8;text-align:left;padding:8px}td{padding:7px 8px;border-top:1px solid #ffffff14}.tw{overflow-x:auto}"
    p = "churn_ai_management_report.html"; open(p, "w", encoding="utf-8").write("<html><head><meta charset='utf-8'><title>CHURN / AI Report</title><style>" + css + "</style></head><body>" + "".join(parts) + "</body></html>")
    return p, note("Report generated. It contains pseudonymous customer IDs: share it only according to your data policy.", "ok")

def _source():
    try:
        for s in (get_ipython().user_ns.get("In") or [])[::-1]:
            if "CHURN / AI — Customer Retention" in s and "def main" in s: return s
    except Exception: pass
    return None

def build_project():
    root = "churn_ai_project"
    for d in [root + "/data", root + "/docs"]: os.makedirs(d, exist_ok=True)
    src = _source()
    if src: open(root + "/app.py", "w", encoding="utf-8").write(src)
    SAMPLE.to_csv(root + "/data/sample_subscription_data.csv", index=False)
    open(root + "/requirements.txt", "w").write(REQS); open(root + "/README.md", "w", encoding="utf-8").write(README)
    open(root + "/.gitignore", "w").write(".env\n*.key\nsecrets*\n__pycache__/\n.ipynb_checkpoints/\n*.zip\nchurn_ai_risk_list.csv\nchurn_ai_batch_scores.csv\n")
    open(root + "/docs/METHODOLOGY.md", "w", encoding="utf-8").write(METH)
    z = "churn_ai_project.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, _, fs in os.walk(root):
            for f in fs: zf.write(os.path.join(dp, f))
    return z, bool(src)

# ------------------------------ UI ------------------------------
CSS = """
.gradio-container{background:#101722!important;color:#F3F7FA!important;font-family:Inter,system-ui,-apple-system,sans-serif!important;max-width:1320px!important}
.hdr{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;padding:26px 30px;border-radius:22px;background:linear-gradient(135deg,#18212D,#12403f 70%,#42C9B455);border:1px solid #42C9B444;margin-bottom:12px;box-shadow:0 10px 30px #0006}
.brand{display:flex;gap:16px;align-items:center}.logo{width:56px;height:56px;border-radius:16px;background:linear-gradient(135deg,#42C9B4,#7BE1C8);color:#101722;font-weight:800;font-size:28px;display:flex;align-items:center;justify-content:center}
.hdr h1{margin:0;font-size:34px;letter-spacing:2px;color:#F3F7FA}.hdr h1 span{color:#42C9B4}.hdr p{margin:4px 0 0;color:#7BE1C8;font-size:16px}
.badges span{display:inline-block;margin:3px;padding:5px 12px;border-radius:99px;background:#ffffff12;border:1px solid #ffffff1f;color:#7BE1C8;font-size:12px}
.kg{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:10px 0}
.kc{background:linear-gradient(180deg,#1d2a38,#18212D);border:1px solid #ffffff14;border-radius:16px;padding:16px;box-shadow:0 6px 18px #0004;animation:fi .5s ease both}
@keyframes fi{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.ki{font-size:22px}.kl{color:#9DADBC;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-top:4px}.kv{font-size:27px;font-weight:700;color:#F3F7FA;margin:4px 0}.kd{color:#9DADBC;font-size:12px}
.note{background:#18212D;padding:10px 14px;border-radius:10px;margin:8px 0;color:#F3F7FA;font-size:14px;line-height:1.5}
.sec{font-size:18px;font-weight:700;color:#7BE1C8;margin:18px 0 6px;padding-left:10px;border-left:4px solid #42C9B4}
.tw{overflow-x:auto;border-radius:14px;border:1px solid #ffffff14;margin:8px 0}table{border-collapse:collapse;width:100%;background:#18212D;font-size:13px}
th{background:#22303f;color:#7BE1C8;text-align:left;padding:10px;white-space:nowrap}td{padding:9px 10px;border-top:1px solid #ffffff10;color:#F3F7FA}tr:hover td{background:#42C9B414}
.pill{padding:3px 10px;border-radius:99px;font-weight:600;white-space:nowrap;font-size:12px}
button.primary{background:linear-gradient(90deg,#42C9B4,#7BE1C8)!important;color:#101722!important;border:none!important;font-weight:700!important}
@media(max-width:640px){.hdr h1{font-size:24px}.kv{font-size:22px}.hdr{padding:18px}}
"""
HEADER = ("<div class='hdr'><div class='brand'><div class='logo'>C/</div><div><h1>CHURN <span>/ AI</span></h1><p>Predict risk. Understand behavior. Retain intelligently.</p></div></div>"
          "<div class='badges'><span>Association ≠ causation</span><span>Hypothetical effects until A/B tested</span><span>Core analytics need no API key</span></div></div>")

def build_ui(zip_path, sample_path):
    with gr.Blocks(css=CSS, title="CHURN / AI", theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate")) as demo:
        gr.HTML(HEADER)
        with gr.Accordion("📂 Data & business assumptions (demo data is FICTIONAL and illustrative)", open=True):
            with gr.Row():
                up = gr.File(label="Upload customer CSV (optional; leave empty for the synthetic demo)", file_types=[".csv"])
                with gr.Column():
                    mp = gr.Textbox(label="Column mapping (only if auto-detect fails)", placeholder="churned=cancelled, monthly_fee=price, tenure_months=months", lines=2)
                    snap = gr.Textbox(label="Snapshot date (optional, YYYY-MM-DD; derives tenure / renewal timing from dates)")
                    gr.File(value=sample_path, label="⬇ Sample dataset (synthetic)", interactive=False)
            with gr.Row():
                mg = gr.Slider(.05, .95, .6, step=.01, label="Contribution margin"); hz = gr.Slider(1, 36, 12, step=1, label="Value horizon (months)")
                cfp = gr.Number(12, label="Cost per retention contact ($): false-alarm cost"); cfn = gr.Number(None, label="Cost of a missed churner ($): blank = median customer value")
            with gr.Row():
                obj = gr.Radio(OBJS, value=OBJS[0], label="Decision-threshold objective"); auto = gr.Checkbox(True, label="Auto risk cut-offs from data")
            with gr.Row():
                tm_ = gr.Slider(.05, .95, .25, step=.01, label="Moderate ≥ (manual)"); th_ = gr.Slider(.05, .95, .40, step=.01, label="High ≥ (manual)"); tc_ = gr.Slider(.05, .95, .65, step=.01, label="Critical ≥ (manual)")
            btn = gr.Button("⚡ Analyze data & train models", variant="primary"); status = gr.HTML(note("Loading the fictional demo dataset and training models (1–3 minutes)…", "info"))
        with gr.Tabs():
            with gr.Tab("A · Executive Overview"):
                kpi = gr.HTML()
                with gr.Row():
                    ex1 = gr.Plot(); ex2 = gr.Plot()
                ex3 = gr.Plot(); prio = gr.HTML()
            with gr.Tab("B · Churn Risk Intelligence"):
                with gr.Row():
                    rk_cat = gr.CheckboxGroup(RISKS, value=["Critical", "High"], label="Risk category")
                    rk_plan = gr.Dropdown(["All"], value="All", label="Plan"); rk_seg = gr.Dropdown(["All"], value="All", label="Segment")
                with gr.Row():
                    rk_tmin = gr.Number(None, label="Tenure min (months)"); rk_tmax = gr.Number(None, label="Tenure max (months)"); rk_q = gr.Textbox(label="Search customer ID")
                    rk_sort = gr.Dropdown(["Priority (expected net benefit)", "Risk score", "Monthly fee", "Expected MRR exposure"], value="Priority (expected net benefit)", label="Sort by")
                    rk_n = gr.Slider(10, 500, 100, step=10, label="Rows to show")
                rk_btn = gr.Button("🔎 Apply filters", variant="primary"); rk_info = gr.HTML()
                rk_df = gr.Dataframe(interactive=False, wrap=True); rk_file = gr.File(label="⬇ Download this list (CSV, pseudonymous IDs only)", interactive=False)
                with gr.Accordion("📦 Batch prediction for new customers + input drift check", open=False):
                    bt_file = gr.File(label="Upload a CSV to score (same columns / mapping as the training data)", file_types=[".csv"]); bt_btn = gr.Button("Score file", variant="primary")
                    bt_html = gr.HTML(); bt_df = gr.Dataframe(interactive=False); bt_out = gr.File(label="⬇ Download scored file", interactive=False)
            with gr.Tab("C · Behavioral Patterns"):
                pat_dd = gr.Dropdown(label="Explore a feature or segment dimension"); pat_fig = gr.Plot(); pat_tbl = gr.HTML()
                with gr.Row():
                    coh_fig = gr.Plot(); trend_fig = gr.Plot()
            with gr.Tab("D · Customer Segmentation"):
                with gr.Row():
                    sg_onb = gr.Slider(1, 12, 3, step=1, label="Onboarding window (months)"); sg_dec = gr.Slider(10, 80, 30, step=5, label="Declining usage: drop ≥ (%)")
                    sg_pf = gr.Slider(1, 5, 2, step=1, label="Payment failures ≥"); sg_unr = gr.Slider(1, 5, 2, step=1, label="Unresolved complaints ≥"); sg_hv = gr.Slider(.5, .95, .75, step=.05, label="High-value = fee quantile ≥")
                sg_btn = gr.Button("Apply segment rules", variant="primary"); seg_html = gr.HTML(); seg_fig = gr.Plot()
                gr.Markdown("#### Optional: K-means clustering (exploratory; clusters are not objectively correct customer types)")
                with gr.Row():
                    cl_k = gr.Slider(2, 8, 4, step=1, label="Number of clusters"); cl_btn = gr.Button("Run clustering", variant="primary")
                cl_html = gr.HTML(); cl_fig = gr.Plot()
            with gr.Tab("E · Individual Customer Profiles"):
                cust_dd = gr.Dropdown(label="Customer ID (top 300 by priority listed; you can also type any ID)", allow_custom_value=True)
                prof_html = gr.HTML(); prof_fig = gr.Plot()
                with gr.Accordion("🧪 What-if prediction for a customer (leave unknown fields blank)", open=False):
                    with gr.Row():
                        ind_plan = gr.Dropdown(label="Plan", allow_custom_value=True); ind_con = gr.Dropdown(label="Contract", allow_custom_value=True)
                        ind_ten = gr.Number(None, label="Tenure (months)"); ind_fee = gr.Number(None, label="Monthly fee")
                    with gr.Row():
                        ind_pf = gr.Number(None, label="Payment failures"); ind_dsl = gr.Number(None, label="Days since last login"); ind_log = gr.Number(None, label="Logins (30 days)"); ind_chg = gr.Number(None, label="Usage change %")
                    with gr.Row():
                        ind_fu = gr.Number(None, label="Features used"); ind_tk = gr.Number(None, label="Support tickets"); ind_un = gr.Number(None, label="Unresolved complaints"); ind_cs = gr.Number(None, label="Satisfaction (1-10)")
                    ind_btn = gr.Button("Predict", variant="primary"); ind_html = gr.HTML(); ind_fig = gr.Plot()
            with gr.Tab("F · Retention Strategy Studio"):
                with gr.Row():
                    strat_dd = gr.Dropdown(label="Customer ID", allow_custom_value=True); st_btn = gr.Button("Generate recommendation", variant="primary")
                st_html = gr.HTML(); st_msg = gr.Textbox(label="Draft message (editable; the app never sends anything)", lines=10, interactive=True)
                msg_btn = gr.Button("Export edited message"); msg_file = gr.File(label="Message file", interactive=False)
                gr.Markdown("#### Retention plan for the highest-priority customers (cells are editable before export)")
                with gr.Row():
                    plan_n = gr.Slider(5, 100, 25, step=5, label="Customers"); plan_seg = gr.Dropdown(["All"] + ALLSEG, value="All", label="Segment"); plan_btn = gr.Button("Build plan", variant="primary")
                plan_df = gr.Dataframe(interactive=True, wrap=True); exp_btn = gr.Button("Export edited plan"); plan_file = gr.File(label="Plan file", interactive=False)
            with gr.Tab("G · Retention ROI Simulator"):
                gr.Markdown("All effects are **hypothetical assumptions**. Observational churn data cannot show how many customers an action will save; confirm with A/B or holdout tests.")
                with gr.Row():
                    roi_thr = gr.Slider(.05, .95, .30, step=.01, label="Minimum risk to target"); roi_cap = gr.Number(300, label="Contact capacity (customers)"); roi_bud = gr.Number(0, label="Budget ($, 0 = no cap)")
                with gr.Row():
                    roi_pos = gr.Checkbox(True, label="Only customers with positive expected net benefit"); roi_scale = gr.Slider(.25, 2, 1, step=.05, label="Effectiveness multiplier")
                strat_tbl = gr.Dataframe(value=DEFAULT_STRATS.copy(), datatype=["str", "number", "number"], interactive=True, label="Editable assumptions: cost per customer ($) and assumed effectiveness (0-1)")
                roi_btn = gr.Button("▶ Run simulation", variant="primary"); roi_html = gr.HTML()
                with gr.Row():
                    roi_f1 = gr.Plot(); roi_f2 = gr.Plot()
            with gr.Tab("H · Model Performance & Explainability"):
                mod_html = gr.HTML()
                with gr.Row():
                    m_roc = gr.Plot(); m_pr = gr.Plot()
                with gr.Row():
                    m_cm = gr.Plot(); m_dist = gr.Plot()
                with gr.Row():
                    m_cal = gr.Plot(); m_thr = gr.Plot()
                m_imp = gr.Plot(); mod_tbl = gr.HTML()
            with gr.Tab("I · Data Quality & Methodology"):
                dq_html = gr.HTML(); tests_html = gr.HTML(); gr.Markdown(METH); gr.Markdown(GITHUB)
                gr.File(value=zip_path, label="⬇ Download GitHub-ready project (ZIP)", interactive=False)
                rep_btn = gr.Button("📄 Generate management report", variant="primary"); rep_msg = gr.HTML(); rep_file = gr.File(label="⬇ Management report (HTML)", interactive=False)
            with gr.Tab("J · AI Retention Agent"):
                gr.Markdown("A tool-using agent: every number comes from computed tools (summary, customer lookup, segments, explanation, recommendation, revenue exposure, simulation, report). Without an API key it answers deterministically; with a key an LLM phrases the answer and its numbers are checked against the tool results.")
                ag_q = gr.Textbox(label="Ask a question", placeholder="Which segments are most vulnerable?  /  Explain CU-10234  /  How much MRR is at risk?")
                with gr.Row():
                    ag_key = gr.Textbox(label="Anthropic API key (optional, session only, never stored)", type="password"); ag_model = gr.Textbox(label="Model", value="claude-sonnet-5-5")
                ag_btn = gr.Button("Ask agent", variant="primary"); ag_out = gr.Markdown()
        ins = [up, mp, snap, mg, hz, cfp, cfn, obj, auto, tm_, th_, tc_]
        RK_IN = [rk_cat, rk_plan, rk_seg, rk_tmin, rk_tmax, rk_q, rk_sort, rk_n]; RK_OUT = [rk_info, rk_df, rk_file]
        RO_IN = [roi_thr, roi_cap, roi_bud, roi_pos, roi_scale, strat_tbl]; RO_OUT = [roi_html, roi_f1, roi_f2]
        def wire(ev):
            e1 = ev.then(refresh, None, [rk_plan, rk_seg, cust_dd, strat_dd, ind_plan, ind_con, pat_dd, strat_tbl])
            e2 = e1.then(render_exec, None, [kpi, ex1, ex2, ex3, prio])
            e3 = e2.then(render_risk, RK_IN, RK_OUT)
            e4 = e3.then(render_patterns, None, [pat_fig, pat_tbl, coh_fig, trend_fig])
            e5 = e4.then(render_seg, None, [seg_html, seg_fig])
            e6 = e5.then(render_roi, RO_IN, RO_OUT)
            e7 = e6.then(render_model, None, [mod_html, m_roc, m_pr, m_cm, m_dist, m_cal, m_thr, m_imp, mod_tbl])
            return e7.then(render_dq, None, [dq_html, tests_html])
        wire(btn.click(analyze, ins, [status], show_progress="full")); wire(demo.load(analyze, ins, [status]))
        rk_btn.click(render_risk, RK_IN, RK_OUT)
        cust_dd.change(render_profile, cust_dd, [prof_html, prof_fig])
        pat_dd.change(pat_chart, pat_dd, pat_fig)
        sg_btn.click(seg_apply, [sg_onb, sg_dec, sg_pf, sg_unr, sg_hv], [seg_html, seg_fig])
        cl_btn.click(render_cluster, cl_k, [cl_html, cl_fig], show_progress="full")
        st_btn.click(render_strategy, strat_dd, [st_html, st_msg]); msg_btn.click(export_msg, st_msg, msg_file)
        plan_btn.click(render_plan, [plan_n, plan_seg], plan_df); exp_btn.click(export_plan, plan_df, plan_file)
        ind_btn.click(predict_one, [ind_plan, ind_con, ind_ten, ind_fee, ind_pf, ind_dsl, ind_log, ind_chg, ind_fu, ind_tk, ind_un, ind_cs], [ind_html, ind_fig])
        roi_btn.click(render_roi, RO_IN, RO_OUT, show_progress="full")
        bt_btn.click(batch_predict, bt_file, [bt_html, bt_df, bt_out], show_progress="full")
        rep_btn.click(make_report, None, [rep_file, rep_msg])
        ag_btn.click(agent_answer, [ag_q, ag_key, ag_model], ag_out); ag_q.submit(agent_answer, [ag_q, ag_key, ag_model], ag_out)
    return demo

def main():
    zip_path, saved = build_project(); sample_path = "churn_ai_sample_data.csv"; SAMPLE.to_csv(sample_path, index=False)
    print("=" * 78); print("CHURN / AI starting. Demo data is FICTIONAL and illustrative. Core analytics need no API key.")
    print("Running self-tests:"); TESTS[:] = run_selftests(verbose=True)
    demo = build_ui(zip_path, sample_path)
    print("Project ZIP:", zip_path, "| app.py included:", "yes" if saved else "NO (paste this cell into app.py yourself)")
    print("First load trains 3 models (1-3 minutes); tabs fill in automatically when it finishes.")
    print("Keys: use Colab Secrets (ANTHROPIC_API_KEY). The public Gradio link is open to anyone: don't share it with real data loaded.")
    print("=" * 78)
    demo.queue().launch(debug=False, show_error=True)

main()