import json
import streamlit as st
import pandas as pd
import altair as alt

# --- データ読み込み ---
with open("filtered_keywords_acl_2025.json", "r", encoding="utf-8") as f:
    keyword_scores = json.load(f)

with open("keyword_to_titles.json", "r", encoding="utf-8") as f:
    keyword_to_titles = json.load(f)

# データ整形
df = pd.DataFrame(list(keyword_scores.items()), columns=["キーワード", "スコア"])
df = df.sort_values(by="スコア", ascending=False).reset_index(drop=True)

# --- UI ---
st.title("📘 ACL 2025: キーワードと関連論文ビューア")

query = st.text_input("🔍 キーワード検索", "")
min_score, max_score = st.slider(
    "スコア範囲",
    float(df["スコア"].min()),
    float(df["スコア"].max()),
    (float(df["スコア"].min()), float(df["スコア"].max()))
)

# --- フィルタ ---
filtered_df = df[
    df["キーワード"].str.contains(query, case=False, na=False) &
    df["スコア"].between(min_score, max_score)
].reset_index(drop=True)

# --- スコア棒グラフ ---
st.markdown("### 🔢 キーワード別 TF-IDF")
bar_chart = alt.Chart(filtered_df).mark_bar().encode(
    x=alt.X("キーワード:N", sort="-y", title="キーワード"),
    y=alt.Y("スコア:Q", title="スコア"),
    tooltip=["キーワード", "スコア"]
).properties(width=800, height=400)
st.altair_chart(bar_chart, use_container_width=True)

# --- 2段階トグルリスト ---
st.markdown("### 📚 キーワード一覧")

with st.expander("🔽 キーワード一覧を表示"):
    keyword = st.selectbox("キーワードを選択", filtered_df["キーワード"])

    st.markdown(f"**🔍 「{keyword}」に関連する論文タイトル：**")
    papers = keyword_to_titles.get(keyword, [])
    if not papers:
        st.write("関連論文は見つかりませんでした。")
    else:
        for paper in papers:
            if isinstance(paper, dict) and "title" in paper:
                st.markdown(f"- {paper['title']}")
            else:
                st.markdown(f"- {paper}")
