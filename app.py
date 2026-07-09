from __future__ import annotations

import html
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


# =========================================
# 基本設定
# =========================================
st.set_page_config(
    page_title="AIコメントアプリ",
    page_icon="📝",
    layout="wide",
)

LIFE_COLUMNS = [
    "病気欠席数",
    "事故欠席数",
    "遅刻数",
    "早退数",
    "忌引等数",
    "出席停止数",
    "保健室利用数",
    "心の天気晴れ数",
    "心の天気曇り数",
    "心の天気雨数",
]

LEARNING_COLUMNS = [
    "総学習時間",
    "解答数",
    "正解率",
]

ANALYSIS_MODES = [
    "過去1週間の傾向",
    "過去1か月の傾向",
    "前週との比較",
    "前月との比較",
]

COMMENT_VIEW_MODES = [
    "1枠：汎用的なコメント",
    "2枠：生活の様子コメント＋学習の様子コメント",
]

COMMENT_PURPOSES = [
    "汎用的なコメント",
    "不登校を防ぐためのコメント",
    "異常値を重視したコメント",
]


@dataclass
class PeriodSetting:
    current_start: pd.Timestamp
    current_end: pd.Timestamp
    previous_start: Optional[pd.Timestamp] = None
    previous_end: Optional[pd.Timestamp] = None

    @property
    def has_previous(self) -> bool:
        return self.previous_start is not None and self.previous_end is not None


def read_csv_japanese_safe(path: Path) -> pd.DataFrame:
    """日本語CSVの文字化けを避けるため、複数の文字コードを順に試す。"""
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
    last_error: Optional[Exception] = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as e:
            last_error = e

    raise RuntimeError(f"CSVの読み込みに失敗しました。文字コードを確認してください。詳細: {last_error}")


def load_sample_data() -> pd.DataFrame:
    sample_path = Path(__file__).parent / "sample_data.csv"
    return read_csv_japanese_safe(sample_path)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """必要な列を整え、日付・数値型を扱いやすくする。"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "日付" not in df.columns:
        raise ValueError("データに『日付』列が必要です。")
    if "児童ID" not in df.columns:
        raise ValueError("データに『児童ID』列が必要です。")

    df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    df = df.dropna(subset=["日付"])

    # 足りない指標列は0で補完。試作段階ではエラーにせず動かす。
    for col in LIFE_COLUMNS + LEARNING_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["児童ID"] = df["児童ID"].astype(str)
    return df.sort_values(["児童ID", "日付"]).reset_index(drop=True)


def determine_period(df: pd.DataFrame, analysis_mode: str) -> PeriodSetting:
    """評価断面に応じて、対象期間と比較期間を決める。"""
    max_date = pd.Timestamp(df["日付"].max()).normalize()

    if analysis_mode == "過去1週間の傾向":
        return PeriodSetting(max_date - pd.Timedelta(days=6), max_date)

    if analysis_mode == "過去1か月の傾向":
        return PeriodSetting(max_date - pd.Timedelta(days=29), max_date)

    if analysis_mode == "前週との比較":
        return PeriodSetting(
            current_start=max_date - pd.Timedelta(days=6),
            current_end=max_date,
            previous_start=max_date - pd.Timedelta(days=13),
            previous_end=max_date - pd.Timedelta(days=7),
        )

    if analysis_mode == "前月との比較":
        # 試作では「直近30日」と「その前30日」の比較とする
        return PeriodSetting(
            current_start=max_date - pd.Timedelta(days=29),
            current_end=max_date,
            previous_start=max_date - pd.Timedelta(days=59),
            previous_end=max_date - pd.Timedelta(days=30),
        )

    return PeriodSetting(max_date - pd.Timedelta(days=6), max_date)


def filter_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["日付"] >= start) & (df["日付"] <= end)].copy()


def aggregate(df: pd.DataFrame, life_columns: List[str], learning_columns: List[str]) -> Dict[str, float]:
    """コメント生成に必要な集計値を作る。未選択の指標はコメント判定に使わない。"""
    result: Dict[str, float] = {"記録日数": float(len(df))}

    for col in LIFE_COLUMNS:
        if col in life_columns:
            result[col] = float(df[col].sum()) if col in df.columns else 0.0
        else:
            result[col] = 0.0

    if "総学習時間" in learning_columns:
        total_study = float(df["総学習時間"].sum()) if "総学習時間" in df.columns else 0.0
    else:
        total_study = 0.0

    if "解答数" in learning_columns:
        total_answers = float(df["解答数"].sum()) if "解答数" in df.columns else 0.0
    else:
        total_answers = 0.0

    # 正解率は、正解率が選択されている場合のみ使う。
    if "正解率" in learning_columns and "正解率" in df.columns and len(df) > 0:
        if "解答数" in df.columns and float(df["解答数"].sum()) > 0:
            weighted_correct_rate = float((df["正解率"] * df["解答数"]).sum() / df["解答数"].sum())
        else:
            weighted_correct_rate = float(df["正解率"].mean())
    else:
        weighted_correct_rate = 0.0

    result["総学習時間"] = total_study
    result["解答数"] = total_answers
    result["正解率"] = weighted_correct_rate
    result["生活指標選択数"] = float(len(life_columns))
    result["学習指標選択数"] = float(len(learning_columns))
    return result


def diff_message(current: Dict[str, float], previous: Optional[Dict[str, float]], keys: List[str]) -> str:
    if not previous:
        return ""

    increased = []
    decreased = []
    for key in keys:
        cur = current.get(key, 0)
        prev = previous.get(key, 0)
        if cur >= prev + 2:
            increased.append(key)
        elif prev >= cur + 2:
            decreased.append(key)

    if increased:
        return "前期間より" + "・".join(increased[:2]) + "が増えています。"
    if decreased:
        return "前期間より" + "・".join(decreased[:2]) + "は減少しています。"
    return "前期間と比べて大きな変化は見られません。"


def fit_text(text: str, limit: int) -> str:
    """文字数上限にできるだけ収める。"""
    text = text.replace("\n", "").strip()
    if len(text) <= limit:
        return text

    cut = text[:limit]
    for sep in ["。", "、"]:
        idx = cut.rfind(sep)
        if idx >= int(limit * 0.55):
            return cut[: idx + 1]

    return cut.rstrip("、。") + "。"


def generate_life_comment(
    current: Dict[str, float],
    previous: Optional[Dict[str, float]],
    analysis_mode: str,
    purpose: str,
    selected_life_columns: List[str],
    limit: int = 50,
) -> str:
    if not selected_life_columns:
        return fit_text("生活面の指標が選択されていません。", limit)

    absence = current.get("病気欠席数", 0) + current.get("事故欠席数", 0)
    late = current.get("遅刻数", 0)
    leave_early = current.get("早退数", 0)
    infirmary = current.get("保健室利用数", 0)
    cloudy = current.get("心の天気曇り数", 0)
    rainy = current.get("心の天気雨数", 0)

    comparison_keys = [
        col
        for col in ["病気欠席数", "事故欠席数", "遅刻数", "早退数", "保健室利用数", "心の天気雨数"]
        if col in selected_life_columns
    ]
    comparison = diff_message(current, previous, comparison_keys) if "比較" in analysis_mode else ""

    if purpose == "異常値を重視したコメント":
        alerts = []
        if "遅刻数" in selected_life_columns and late >= 2:
            alerts.append("遅刻")
        if "保健室利用数" in selected_life_columns and infirmary >= 2:
            alerts.append("保健室利用")
        if "心の天気雨数" in selected_life_columns and rainy >= 2:
            alerts.append("心の天気の雨")
        if ({"病気欠席数", "事故欠席数"} & set(selected_life_columns)) and absence >= 2:
            alerts.append("欠席")
        if alerts:
            text = "・".join(alerts[:2]) + "が目立ちます。生活リズムや体調面の変化に注意が必要です。"
        else:
            text = "選択した生活指標では、特に目立つ異常値は見られません。"
        return fit_text(text, limit)

    if purpose == "不登校を防ぐためのコメント":
        if absence + late + leave_early + infirmary + rainy >= 4:
            text = "生活面に複数の変化が見られます。無理のない声かけで様子を確認します。"
        elif late + infirmary + rainy >= 2:
            text = "遅刻や体調面の変化が見られます。日常の声かけを通じて見守ります。"
        else:
            text = "選択した生活指標では大きく乱れていません。普段の様子を見守ります。"
        return fit_text(text, limit)

    if comparison:
        text = comparison + "生活面の変化を継続して確認します。"
    elif absence + late + leave_early + infirmary + rainy >= 4:
        text = "遅刻や保健室利用などが見られます。生活リズムや体調面の変化に注意が必要です。"
    elif cloudy + rainy >= 3:
        text = "心の天気に曇りや雨が見られます。気持ちの変化を丁寧に見守ります。"
    else:
        text = "選択した生活指標では、大きな乱れは見られません。安定した様子です。"
    return fit_text(text, limit)


def generate_learning_comment(
    current: Dict[str, float],
    previous: Optional[Dict[str, float]],
    analysis_mode: str,
    purpose: str,
    selected_learning_columns: List[str],
    limit: int = 50,
) -> str:
    if not selected_learning_columns:
        return fit_text("学習面の指標が選択されていません。", limit)

    study = current.get("総学習時間", 0)
    answers = current.get("解答数", 0)
    rate = current.get("正解率", 0)

    if previous:
        prev_study = previous.get("総学習時間", 0)
        prev_answers = previous.get("解答数", 0)
        prev_rate = previous.get("正解率", 0)
    else:
        prev_study = prev_answers = prev_rate = 0

    if "比較" in analysis_mode and previous:
        if "総学習時間" in selected_learning_columns and "解答数" in selected_learning_columns and study < prev_study * 0.8 and answers < prev_answers * 0.8:
            text = "前期間より学習量が減っています。取り組み状況の変化を確認します。"
        elif "正解率" in selected_learning_columns and rate + 0.05 < prev_rate:
            text = "前期間より正解率が下がっています。理解度の変化を確認します。"
        elif ("総学習時間" in selected_learning_columns and study > prev_study * 1.2) or ("解答数" in selected_learning_columns and answers > prev_answers * 1.2):
            text = "前期間より学習量が増えています。取り組みの継続を見守ります。"
        else:
            text = "前期間と比べて、選択した学習指標に大きな変化は見られません。"
        return fit_text(text, limit)

    if purpose == "異常値を重視したコメント":
        if (("解答数" in selected_learning_columns and answers == 0) or ("総学習時間" in selected_learning_columns and study == 0)):
            text = "学習記録が少ない日があります。取り組み状況の確認が必要です。"
        elif "正解率" in selected_learning_columns and rate < 0.70:
            text = "正解率が低めです。理解が難しい単元がないか確認します。"
        else:
            text = "選択した学習指標では、特に目立つ異常値は見られません。"
        return fit_text(text, limit)

    if purpose == "不登校を防ぐためのコメント":
        if ("総学習時間" in selected_learning_columns and study < 180) or ("解答数" in selected_learning_columns and answers < 80):
            text = "学習量が少なめです。無理のない範囲で取り組みを確認します。"
        elif "正解率" in selected_learning_columns and rate < 0.75:
            text = "正解率に課題が見られます。理解度に応じた支援を検討します。"
        else:
            text = "学習への取り組みは一定程度見られます。継続して見守ります。"
        return fit_text(text, limit)

    if ("総学習時間" in selected_learning_columns and study == 0) and ("解答数" in selected_learning_columns and answers == 0):
        text = "学習記録が確認できません。対象期間の取り組み状況を確認します。"
    elif "正解率" in selected_learning_columns and rate < 0.70:
        text = "学習には取り組んでいますが、正解率は低めです。理解度を確認します。"
    elif ("総学習時間" in selected_learning_columns and study >= 250) or ("解答数" in selected_learning_columns and answers >= 120):
        text = "学習時間や解答数は十分に見られます。取り組みは継続できています。"
    else:
        text = "選択した学習指標では、一定程度の取り組みが見られます。"
    return fit_text(text, limit)


def generate_generic_comment(
    current: Dict[str, float],
    previous: Optional[Dict[str, float]],
    analysis_mode: str,
    purpose: str,
    selected_life_columns: List[str],
    selected_learning_columns: List[str],
    limit: int = 100,
) -> str:
    life = generate_life_comment(current, previous, analysis_mode, purpose, selected_life_columns, limit=55)
    learning = generate_learning_comment(current, previous, analysis_mode, purpose, selected_learning_columns, limit=55)
    text = f"{life}{learning}"
    return fit_text(text, limit)


def generate_comments(
    current: Dict[str, float],
    previous: Optional[Dict[str, float]],
    analysis_mode: str,
    view_mode: str,
    purpose: str,
    selected_life_columns: List[str],
    selected_learning_columns: List[str],
) -> Dict[str, str]:
    if view_mode.startswith("1枠"):
        return {
            "汎用的なコメント": generate_generic_comment(current, previous, analysis_mode, purpose, selected_life_columns, selected_learning_columns, limit=100),
            "生活の様子コメント": "",
            "学習の様子コメント": "",
        }

    return {
        "汎用的なコメント": "",
        "生活の様子コメント": generate_life_comment(current, previous, analysis_mode, purpose, selected_life_columns, limit=50),
        "学習の様子コメント": generate_learning_comment(current, previous, analysis_mode, purpose, selected_learning_columns, limit=50),
    }


def render_comment_box(title: str, text: str, box_type: str, animate: bool = True) -> None:
    """AIが文章を生成しているように、コメントを1文字ずつ表示する。"""
    placeholder = st.empty()

    def box_html(body: str) -> str:
        safe_type = html.escape(box_type)
        return f"""
        <div class="comment-card comment-card-{safe_type}">
            <div class="comment-card-title">{html.escape(title)}</div>
            <div class="comment-card-body">{html.escape(body)}</div>
        </div>
        """

    if not animate:
        placeholder.markdown(box_html(text), unsafe_allow_html=True)
        return

    display_text = ""
    for ch in text:
        display_text += ch
        placeholder.markdown(box_html(display_text + "▌"), unsafe_allow_html=True)
        time.sleep(0.018)
    placeholder.markdown(box_html(text), unsafe_allow_html=True)


def render_comment_page(comments: Dict[str, str], view_mode: str) -> None:
    if view_mode.startswith("1枠"):
        render_comment_box("汎用的なコメント", comments["汎用的なコメント"], box_type="generic")
        return

    c1, c2 = st.columns(2)
    with c1:
        render_comment_box("生活の様子コメント", comments["生活の様子コメント"], box_type="life")
    with c2:
        render_comment_box("学習の様子コメント", comments["学習の様子コメント"], box_type="learning")


def render_data_page(current_df: pd.DataFrame, period: PeriodSetting, selected_life_columns: List[str], selected_learning_columns: List[str]) -> None:
    st.subheader("対象期間のデータ")
    st.caption(
        f"対象期間：{period.current_start:%Y/%m/%d} 〜 {period.current_end:%Y/%m/%d}"
    )
    selected_columns = ["日付"] + selected_life_columns + selected_learning_columns
    selected_columns = [col for col in selected_columns if col in current_df.columns]
    display_df = current_df[selected_columns].copy()
    display_df["日付"] = display_df["日付"].dt.strftime("%Y/%m/%d")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# =========================================
# スタイル
# =========================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
    }
    .comment-card {
        border-radius: 20px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.10);
        padding: 24px 26px;
        margin-top: 12px;
        min-height: 168px;
    }
    .comment-card-generic {
        background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
        border: 2px solid #60a5fa;
    }
    .comment-card-life {
        background: linear-gradient(135deg, #fff7ed 0%, #ffffff 100%);
        border: 2px solid #fb923c;
    }
    .comment-card-learning {
        background: linear-gradient(135deg, #f5f3ff 0%, #ffffff 100%);
        border: 2px solid #a78bfa;
    }
    .comment-card-title {
        font-size: 19px;
        font-weight: 800;
        margin-bottom: 14px;
        color: #111827;
    }
    .comment-card-body {
        font-size: 25px;
        line-height: 1.85;
        font-weight: 700;
        color: #111827;
        word-break: break-word;
    }
    .stButton > button {
        border-radius: 999px;
        padding: 0.6rem 1.2rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================
# 画面
# =========================================
st.title("AIコメントアプリ")

try:
    raw_df = load_sample_data()
    df = normalize_dataframe(raw_df)
except Exception as e:
    st.error(str(e))
    st.stop()

# 今回の試作は1人分のサンプルデータを前提にする。
student_ids = sorted(df["児童ID"].unique().tolist())
selected_student = student_ids[0]
student_df = df[df["児童ID"] == selected_student]

if "show_data_page" not in st.session_state:
    st.session_state.show_data_page = False

with st.sidebar:
    st.header("AIコメント定義")
    analysis_mode = st.radio("評価するデータ断面", ANALYSIS_MODES)
    view_mode = st.radio("コメント枠", COMMENT_VIEW_MODES)
    purpose = st.radio("コメントの方針", COMMENT_PURPOSES)

    st.markdown("---")
    st.subheader("使う指標")
    selected_life_columns = st.multiselect(
        "生活の様子コメントに使う指標",
        LIFE_COLUMNS,
        default=LIFE_COLUMNS,
    )
    selected_learning_columns = st.multiselect(
        "学習の様子コメントに使う指標",
        LEARNING_COLUMNS,
        default=LEARNING_COLUMNS,
    )

period = determine_period(df, analysis_mode)
current_df = filter_period(student_df, period.current_start, period.current_end)
previous_df = filter_period(student_df, period.previous_start, period.previous_end) if period.has_previous else None
current_agg = aggregate(current_df, selected_life_columns, selected_learning_columns)
previous_agg = aggregate(previous_df, selected_life_columns, selected_learning_columns) if previous_df is not None else None
comments = generate_comments(
    current_agg,
    previous_agg,
    analysis_mode,
    view_mode,
    purpose,
    selected_life_columns,
    selected_learning_columns,
)

if st.session_state.show_data_page:
    render_data_page(current_df, period, selected_life_columns, selected_learning_columns)
    if st.button("AIコメントに戻る"):
        st.session_state.show_data_page = False
        st.rerun()
else:
    st.header("A君へのAIコメント")
    st.subheader("コメント枠")
    render_comment_page(comments, view_mode)
    st.markdown("---")
    if st.button("対象期間のデータを表示する"):
        st.session_state.show_data_page = True
        st.rerun()
