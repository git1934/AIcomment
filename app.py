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

COMMENT_TYPES = [
    "定性的なコメント",
    "定量的なコメント",
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



def fmt_count(value: float) -> str:
    """表示用に小数を避けて回数・件数を整える。"""
    return str(int(round(value)))


def fmt_minutes(value: float) -> str:
    return f"{int(round(value))}分"


def fmt_rate(value: float) -> str:
    return f"{value * 100:.0f}%"


def metric_display_name(metric: str) -> str:
    names = {
        "病気欠席数": "病気欠席",
        "事故欠席数": "事故欠席",
        "遅刻数": "遅刻",
        "早退数": "早退",
        "忌引等数": "忌引等",
        "出席停止数": "出席停止",
        "保健室利用数": "保健室利用",
        "心の天気晴れ数": "晴れ",
        "心の天気曇り数": "曇り",
        "心の天気雨数": "雨",
        "総学習時間": "学習時間",
        "解答数": "解答数",
        "正解率": "正解率",
    }
    return names.get(metric, metric)


def metric_value_text(metric: str, value: float) -> str:
    if metric == "総学習時間":
        return fmt_minutes(value)
    if metric == "正解率":
        return fmt_rate(value)
    if metric == "解答数":
        return f"{fmt_count(value)}問"
    return f"{fmt_count(value)}回"


def summarize_selected_totals(values: Dict[str, float], metrics: List[str], max_items: int = 3) -> str:
    """選択指標から、値が大きいものを数値付きで短く並べる。"""
    non_zero = [(m, values.get(m, 0.0)) for m in metrics if values.get(m, 0.0) != 0]
    if not non_zero:
        non_zero = [(m, values.get(m, 0.0)) for m in metrics[:max_items]]
    non_zero.sort(key=lambda x: abs(x[1]), reverse=True)
    parts = [f"{metric_display_name(m)}{metric_value_text(m, v)}" for m, v in non_zero[:max_items]]
    return "、".join(parts)


def summarize_selected_changes(
    current: Dict[str, float],
    previous: Optional[Dict[str, float]],
    metrics: List[str],
    max_items: int = 2,
) -> str:
    """比較モード用に、変化が大きい指標を 旧→新 で短く表現する。"""
    if not previous:
        return summarize_selected_totals(current, metrics, max_items=max_items)

    changes = []
    for metric in metrics:
        cur = current.get(metric, 0.0)
        prev = previous.get(metric, 0.0)
        diff = cur - prev
        if metric == "正解率":
            score = abs(diff) * 100
        else:
            score = abs(diff)
        changes.append((score, metric, prev, cur, diff))

    changes.sort(key=lambda x: x[0], reverse=True)
    meaningful = [item for item in changes if item[0] > 0]
    target = meaningful[:max_items] if meaningful else changes[:max_items]

    parts = []
    for _, metric, prev, cur, _ in target:
        parts.append(
            f"{metric_display_name(metric)}{metric_value_text(metric, prev)}→{metric_value_text(metric, cur)}"
        )
    return "、".join(parts)


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



def qualitative_life_comment(
    current: Dict[str, float],
    previous: Optional[Dict[str, float]],
    analysis_mode: str,
    purpose: str,
    selected_life_columns: List[str],
    limit: int,
) -> str:
    if not selected_life_columns:
        return fit_text("生活面の指標が選択されていません。", limit)

    key_life_metrics = [
        m
        for m in ["病気欠席数", "事故欠席数", "遅刻数", "早退数", "保健室利用数", "心の天気雨数", "心の天気曇り数"]
        if m in selected_life_columns
    ] or selected_life_columns

    if "比較" in analysis_mode:
        base = diff_message(current, previous, key_life_metrics)
        if purpose == "異常値を重視したコメント":
            text = f"{base}変化が目立つ項目を優先して確認してください。"
        elif purpose == "不登校を防ぐためのコメント":
            text = f"{base}登校状況や生活リズムの変化を確認してください。"
        else:
            text = f"{base}生活面の変化を継続して確認してください。"
        return fit_text(text, limit)

    absence = current.get("病気欠席数", 0) + current.get("事故欠席数", 0)
    late = current.get("遅刻数", 0)
    leave_early = current.get("早退数", 0)
    infirmary = current.get("保健室利用数", 0)
    cloudy = current.get("心の天気曇り数", 0)
    rainy = current.get("心の天気雨数", 0)
    total_signals = absence + late + leave_early + infirmary + cloudy + rainy

    if purpose == "異常値を重視したコメント":
        if total_signals >= 4:
            text = "対象期間では生活面に気になる変化があります。目立つ項目を優先して確認してください。"
        else:
            text = "対象期間では生活面の大きな変化は少ない状況です。"
    elif purpose == "不登校を防ぐためのコメント":
        if absence + late + infirmary + rainy >= 4:
            text = "生活面に変化が見られます。登校状況や体調面を早めに確認してください。"
        else:
            text = "生活面は比較的落ち着いています。日々の様子を継続して確認してください。"
    else:
        if total_signals >= 4:
            text = "対象期間では生活面にやや変化が見られます。継続して様子を確認してください。"
        else:
            text = "対象期間では生活面に大きな乱れは少ない状況です。"
    return fit_text(text, limit)


def qualitative_learning_comment(
    current: Dict[str, float],
    previous: Optional[Dict[str, float]],
    analysis_mode: str,
    purpose: str,
    selected_learning_columns: List[str],
    limit: int,
) -> str:
    if not selected_learning_columns:
        return fit_text("学習面の指標が選択されていません。", limit)

    if "比較" in analysis_mode and previous:
        decreased = []
        increased = []
        for metric in selected_learning_columns:
            cur = current.get(metric, 0.0)
            prev = previous.get(metric, 0.0)
            if metric == "正解率":
                if cur <= prev - 0.05:
                    decreased.append(metric)
                elif cur >= prev + 0.05:
                    increased.append(metric)
            else:
                if cur < prev:
                    decreased.append(metric)
                elif cur > prev:
                    increased.append(metric)

        if decreased:
            trend = "前期間より学習面に低下傾向が見られます。"
        elif increased:
            trend = "前期間より学習面に改善傾向が見られます。"
        else:
            trend = "前期間と比べて学習面に大きな変化は見られません。"

        if purpose == "異常値を重視したコメント":
            text = f"{trend}変化の大きい項目を確認してください。"
        elif purpose == "不登校を防ぐためのコメント":
            text = f"{trend}学習意欲や負担感を確認してください。"
        else:
            text = f"{trend}学習状況の推移を確認してください。"
        return fit_text(text, limit)

    study = current.get("総学習時間", 0)
    answers = current.get("解答数", 0)
    rate = current.get("正解率", 0)

    if purpose == "異常値を重視したコメント":
        if study == 0 or answers == 0 or ("正解率" in selected_learning_columns and rate < 0.70):
            text = "対象期間では学習面に気になる変化があります。未実施や低下を確認してください。"
        else:
            text = "対象期間では学習面に大きな異常は少ない状況です。"
    elif purpose == "不登校を防ぐためのコメント":
        text = "学習面の取り組み状況と負担感を確認し、必要に応じて声かけをしてください。"
    elif "正解率" in selected_learning_columns and rate < 0.70:
        text = "対象期間では学習面にやや課題が見られます。理解状況を確認してください。"
    else:
        text = "対象期間では学習面の取り組みが見られます。継続して様子を確認してください。"
    return fit_text(text, limit)

def generate_life_comment(
    current: Dict[str, float],
    previous: Optional[Dict[str, float]],
    analysis_mode: str,
    purpose: str,
    selected_life_columns: List[str],
    comment_type: str,
    limit: int = 50,
) -> str:
    if comment_type == "定性的なコメント":
        return qualitative_life_comment(current, previous, analysis_mode, purpose, selected_life_columns, limit)

    if not selected_life_columns:
        return fit_text("生活面の指標が選択されていません。", limit)

    absence_metrics = [m for m in ["病気欠席数", "事故欠席数"] if m in selected_life_columns]
    key_life_metrics = [
        m
        for m in ["病気欠席数", "事故欠席数", "遅刻数", "早退数", "保健室利用数", "心の天気雨数", "心の天気曇り数"]
        if m in selected_life_columns
    ]
    if not key_life_metrics:
        key_life_metrics = selected_life_columns

    absence = sum(current.get(m, 0) for m in absence_metrics)
    late = current.get("遅刻数", 0)
    leave_early = current.get("早退数", 0)
    infirmary = current.get("保健室利用数", 0)
    cloudy = current.get("心の天気曇り数", 0)
    rainy = current.get("心の天気雨数", 0)

    if "比較" in analysis_mode:
        summary = summarize_selected_changes(current, previous, key_life_metrics, max_items=2)
        if purpose == "異常値を重視したコメント":
            text = f"前期間比で{summary}。変化の大きい項目を確認してください。"
        elif purpose == "不登校を防ぐためのコメント":
            text = f"前期間比で{summary}。登校状況や体調面を確認してください。"
        else:
            text = f"前期間比で{summary}。生活面の変化を確認してください。"
        return fit_text(text, limit)

    summary = summarize_selected_totals(current, key_life_metrics, max_items=3)
    if purpose == "異常値を重視したコメント":
        text = f"対象期間は{summary}。数値が目立つ項目を優先して確認してください。"
    elif purpose == "不登校を防ぐためのコメント":
        if absence + late + leave_early + infirmary + rainy >= 4:
            text = f"対象期間は{summary}。生活面の変化に早めに声かけをしてください。"
        else:
            text = f"対象期間は{summary}。普段の様子と合わせて確認してください。"
    else:
        if absence + late + leave_early + infirmary + cloudy + rainy >= 4:
            text = f"対象期間は{summary}。生活面に変化が見られます。"
        else:
            text = f"対象期間は{summary}。大きな乱れは少ない状況です。"
    return fit_text(text, limit)

def generate_learning_comment(
    current: Dict[str, float],
    previous: Optional[Dict[str, float]],
    analysis_mode: str,
    purpose: str,
    selected_learning_columns: List[str],
    comment_type: str,
    limit: int = 50,
) -> str:
    if comment_type == "定性的なコメント":
        return qualitative_learning_comment(current, previous, analysis_mode, purpose, selected_learning_columns, limit)

    if not selected_learning_columns:
        return fit_text("学習面の指標が選択されていません。", limit)

    study = current.get("総学習時間", 0)
    answers = current.get("解答数", 0)
    rate = current.get("正解率", 0)

    if "比較" in analysis_mode and previous:
        summary = summarize_selected_changes(current, previous, selected_learning_columns, max_items=2)
        prev_study = previous.get("総学習時間", 0)
        prev_answers = previous.get("解答数", 0)
        prev_rate = previous.get("正解率", 0)

        if purpose == "異常値を重視したコメント":
            text = f"前期間比で{summary}。変化の大きい項目を確認してください。"
        elif purpose == "不登校を防ぐためのコメント":
            text = f"前期間比で{summary}。学習意欲や負担感を確認してください。"
        elif (
            ("総学習時間" in selected_learning_columns and study < prev_study)
            or ("解答数" in selected_learning_columns and answers < prev_answers)
            or ("正解率" in selected_learning_columns and rate < prev_rate)
        ):
            text = f"前期間比で{summary}。学習面は低下傾向です。"
        else:
            text = f"前期間比で{summary}。学習面の推移を確認してください。"
        return fit_text(text, limit)

    summary = summarize_selected_totals(current, selected_learning_columns, max_items=3)
    if purpose == "異常値を重視したコメント":
        text = f"対象期間は{summary}。低下や未実施の有無を確認してください。"
    elif purpose == "不登校を防ぐためのコメント":
        text = f"対象期間は{summary}。無理のない学習状況を確認してください。"
    elif ("正解率" in selected_learning_columns and rate < 0.70):
        text = f"対象期間は{summary}。正解率が低めです。"
    else:
        text = f"対象期間は{summary}。学習面の取り組みを確認してください。"
    return fit_text(text, limit)

def generate_generic_comment(
    current: Dict[str, float],
    previous: Optional[Dict[str, float]],
    analysis_mode: str,
    purpose: str,
    selected_life_columns: List[str],
    selected_learning_columns: List[str],
    comment_type: str,
    limit: int = 100,
) -> str:
    """
    1枠表示用の汎用コメントを生成する。

    生活・学習のどちらか一方だけを選んだ場合は、選択されている指標だけで
    コメントを作る。未選択側の「指標が選択されていません」という文言は表示しない。
    """
    has_life = len(selected_life_columns) > 0
    has_learning = len(selected_learning_columns) > 0

    if has_life and has_learning:
        life = generate_life_comment(
            current,
            previous,
            analysis_mode,
            purpose,
            selected_life_columns,
            comment_type,
            limit=55,
        )
        learning = generate_learning_comment(
            current,
            previous,
            analysis_mode,
            purpose,
            selected_learning_columns,
            comment_type,
            limit=55,
        )
        return fit_text(f"{life}{learning}", limit)

    if has_life:
        return generate_life_comment(
            current,
            previous,
            analysis_mode,
            purpose,
            selected_life_columns,
            comment_type,
            limit=limit,
        )

    if has_learning:
        return generate_learning_comment(
            current,
            previous,
            analysis_mode,
            purpose,
            selected_learning_columns,
            comment_type,
            limit=limit,
        )

    return ""


def generate_comments(
    current: Dict[str, float],
    previous: Optional[Dict[str, float]],
    analysis_mode: str,
    view_mode: str,
    purpose: str,
    comment_type: str,
    selected_life_columns: List[str],
    selected_learning_columns: List[str],
) -> Dict[str, str]:
    if view_mode.startswith("1枠"):
        return {
            "汎用的なコメント": generate_generic_comment(current, previous, analysis_mode, purpose, selected_life_columns, selected_learning_columns, comment_type, limit=100),
            "生活の様子コメント": "",
            "学習の様子コメント": "",
        }

    return {
        "汎用的なコメント": "",
        "生活の様子コメント": generate_life_comment(current, previous, analysis_mode, purpose, selected_life_columns, comment_type, limit=50),
        "学習の様子コメント": generate_learning_comment(current, previous, analysis_mode, purpose, selected_learning_columns, comment_type, limit=50),
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


def build_function_requirements_markdown(
    selected_student: str,
    view_mode: str,
    purpose: str,
    comment_type: str,
    analysis_mode: str,
    selected_life_columns: List[str],
    selected_learning_columns: List[str],
    period: PeriodSetting,
    current_df: pd.DataFrame,
    previous_df: Optional[pd.DataFrame],
    comments: Dict[str, str],
) -> str:
    """現在の画面設定を反映した機能要件書をMarkdown形式で生成する。"""

    generated_at = pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y/%m/%d %H:%M")
    current_period = f"{period.current_start:%Y/%m/%d} 〜 {period.current_end:%Y/%m/%d}"

    if period.has_previous:
        previous_period = f"{period.previous_start:%Y/%m/%d} 〜 {period.previous_end:%Y/%m/%d}"
    else:
        previous_period = "なし"

    life_metrics = "\n".join(f"- {metric}" for metric in selected_life_columns) or "- 選択なし"
    learning_metrics = "\n".join(f"- {metric}" for metric in selected_learning_columns) or "- 選択なし"

    comment_sections = []
    for title, comment in comments.items():
        if comment:
            comment_sections.append(f"### {title}\n\n{comment}")

    generated_comments = "\n\n".join(comment_sections) or "生成対象のコメントはありません。"

    comparison_requirement = (
        "対象期間と比較期間を集計し、選択指標ごとの差分をコメントに反映する。"
        if period.has_previous
        else "対象期間のみを集計し、選択指標の傾向をコメントに反映する。"
    )

    return f"""# AIコメント機能要件一覧

## 1. ドキュメント情報

- 出力日時：{generated_at}
- 対象児童ID：{selected_student}
- アプリ名：AIコメント要件一覧出力アプリ
- 出力形式：Markdown

## 2. 機能概要

児童の生活面・学習面の記録データを集計し、画面左側で指定した条件に基づいて、
先生向けのコメントを生成する。

## 3. 現在の画面設定

| 設定項目 | 設定値 |
|---|---|
| コメント枠 | {view_mode} |
| コメントの方針 | {purpose} |
| コメントの種類 | {comment_type} |
| 評価するデータ断面 | {analysis_mode} |
| 対象期間 | {current_period} |
| 比較期間 | {previous_period} |
| 対象期間のデータ件数 | {len(current_df)}件 |
| 比較期間のデータ件数 | {len(previous_df) if previous_df is not None else 0}件 |

## 4. 使用する指標

### 4.1 生活面

{life_metrics}

### 4.2 学習面

{learning_metrics}

## 5. コメント生成要件

1. 画面左側で選択された指標のみを集計対象とする。
2. 未選択の指標はコメント判定および数値表示に使用しない。
3. {comparison_requirement}
4. 「定性的なコメント」では、数値の列挙よりも傾向や確認事項を中心に出力する。
5. 「定量的なコメント」では、対象期間の集計値または前期間との差を含めて出力する。
6. 1枠表示では、生活面・学習面のうち選択されている指標だけを使ってコメントを生成する。
7. 1枠表示で片方の指標が未選択の場合、未選択側の警告文は表示しない。
8. 2枠表示では、「生活の様子コメント」と「学習の様子コメント」を個別に生成する。
9. コメント文は、アプリ自身が行動する表現を避け、「確認してください」「声かけをしてください」など利用者向けの表現とする。
10. コメント文字数は、1枠表示では最大100文字、2枠表示では各最大50文字を基本とする。
"""


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
    view_mode = st.radio("コメント枠", COMMENT_VIEW_MODES)
    purpose = st.radio("コメントの方針", COMMENT_PURPOSES)
    comment_type = st.radio("コメントの種類", COMMENT_TYPES)
    analysis_mode = st.radio("評価するデータ断面", ANALYSIS_MODES)

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
    comment_type,
    selected_life_columns,
    selected_learning_columns,
)

requirements_markdown = build_function_requirements_markdown(
    selected_student=selected_student,
    view_mode=view_mode,
    purpose=purpose,
    comment_type=comment_type,
    analysis_mode=analysis_mode,
    selected_life_columns=selected_life_columns,
    selected_learning_columns=selected_learning_columns,
    period=period,
    current_df=current_df,
    previous_df=previous_df,
    comments=comments,
)

with st.sidebar:
    st.markdown("---")
    st.subheader("機能要件")
    st.download_button(
        label="機能要件一覧を出力する",
        data=requirements_markdown,
        file_name="ai_comment_function_requirements.md",
        mime="text/markdown",
        use_container_width=True,
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
