# app.py
import os
from dotenv import load_dotenv

# ---- 環境変数（APIキー）読み込み ----
load_dotenv()  # .env から OPENAI_API_KEY を読み込み

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ---- LLM 初期化 ----
# gpt-4o-mini を温度0で安定志向に
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---- 役割（専門家）ごとの System メッセージひな形 ----
SYSTEM_TEMPLATES = {
    "教師": (
        "あなたは生徒にわかりやすく教えるプロの教師です。"
        "専門用語はできるだけ噛み砕き、具体例を交えて、箇条書き中心で簡潔に説明してください。"
    ),
    "弁護士": (
        "あなたは依頼者に法的助言を行う弁護士です。"
        "まず結論、その後に根拠法や注意点を整理して、誤解のない表現で説明してください。"
        "法的助言で不確実な点があれば、その旨を明記してください。"
    ),
}

# ---- LLM呼び出し用の関数（要件：引数2つを受け取り、回答文字列を返す） ----
def ask_llm(user_text: str, expert_role: str) -> str:
    """
    入力テキスト(user_text)とラジオボタン選択値(expert_role: '教師' or '弁護士')
    を受け取り、LLMの回答テキストを返す。
    """
    system_prompt = SYSTEM_TEMPLATES.get(expert_role, SYSTEM_TEMPLATES["教師"])
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text.strip())
    ]
    response = llm.invoke(messages)  # LangChainのRunnableインタフェース
    return response.content

# ---- Streamlit UI ----
st.set_page_config(page_title="専門家アシスタント（教師/弁護士）", page_icon="🎓", layout="centered")

# ヘッダーと概要（要件：アプリ概要・操作方法の明示）
st.title("専門家アシスタント（教師 / 弁護士）")
st.caption(
    "入力テキストと専門家の種類を選ぶだけで、LLMが役割に応じた回答を返します。"
    "このアプリは LangChain を用いてプロンプトを組み立てています。"
)

with st.expander("📝 このWebアプリの使い方（概要・操作方法）", expanded=True):
    st.markdown(
        """
**使い方**
1. 下のラジオボタンで「教師」または「弁護士」を選びます。  
2. テキスト入力欄に質問や相談内容を入力します。  
3. **送信**ボタンを押すと、選択した役割に合わせた回答が表示されます。  

**ポイント**
- 「教師」を選ぶと、やさしい言葉と具体例中心の解説になります。  
- 「弁護士」を選ぶと、結論→根拠→注意点の順で、法律の観点からの説明になります。  
- APIキーは `.env` から読み込みます（`OPENAI_API_KEY`）。  
        """
    )

# 専門家選択（ラジオボタン）
expert = st.radio("専門家の種類を選択", options=["教師", "弁護士"], horizontal=True)

# テキスト入力
user_input = st.text_area(
    "入力テキスト（質問や相談内容を入力）",
    placeholder="例）生成AIの基礎を学ぶには何から始めるべき？ / 契約書レビュー時の注意点は？",
    height=140,
)

# 送信ボタン
submit = st.button("送信", type="primary", use_container_width=True)

# 応答表示
if submit:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY が設定されていません。.env を確認してください。")
    elif not user_input.strip():
        st.warning("入力テキストを入力してください。")
    else:
        with st.spinner("LLMに問い合わせ中..."):
            try:
                answer = ask_llm(user_input, expert)
                st.subheader("回答")
                st.write(answer)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                st.info("依存パッケージのバージョンや API キーの設定を確認してください。")
