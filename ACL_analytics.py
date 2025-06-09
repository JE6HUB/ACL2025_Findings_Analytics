import requests
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json
import os

from openai import OpenAI
# OpenAIのAPIクライアントを初期化
client = OpenAI()   
# 


# NLTKのリソースをダウンロード
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. データ収集
url = 'https://2025.aclweb.org/program/find_papers/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# タイトルと要約の抽出（例としてタイトルのみ）
titles = [li.get_text() for li in soup.find_all('li')]

# 2. テキスト前処理
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

#openai用のプロンプトを作成
def make_prompt(i, word):
    return {
        "custom_id": f"request-{i + 1}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": f"次の単語「{word}」は自然言語処理（NLP）の研究分野に関連していますか？Yes または No で答えてください。"
                }
            ],
            "max_tokens": 10
        }
    }

def openai_input(openai_inputs):
    # OpenAI APIにバッチリクエストを送信
    batch_input_file = client.files.create(
        file=open(openai_inputs, "rb"),
        purpose="batch"
    )

    # バッチリクエストを実行

    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Batch request for ACL 2025 keywords",
        }
    )

processed_titles = [preprocess(title) for title in titles]

# 3. TF-IDFによるキーワード抽出
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_titles)
feature_names = vectorizer.get_feature_names_out()
dense = tfidf_matrix.todense()
denselist = dense.tolist()


# キーワードの頻度を集計
keyword_freq = {}
for doc in denselist:
    for word, score in zip(feature_names, doc):
        keyword_freq[word] = keyword_freq.get(word, 0) + score

# 上位キーワードの抽出
sorted_keywords = dict(sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True))
#キーワードをjson形式で保存
with open('top_keywords_acl_2025.json', 'w') as f:
    json.dump(sorted_keywords, f, indent=4)

# 上位20キーワードを抽出
top_keywords = dict(list(sorted_keywords.items())[:20])

openai_inputs = 'openai_inputs.jsonl'

# JSONLとして書き出し
with open(openai_inputs, "w", encoding="utf-8") as f:    
    for i, word in enumerate(sorted_keywords.keys()):
        prompt = make_prompt(i, word)
        json.dump(prompt, f, ensure_ascii=False)
        f.write("\n")

# --- 追加：keyword_to_titles.jsonの生成 ---
keyword_to_titles = {}
for keyword in sorted_keywords.keys():
    keyword_to_titles[keyword] = [
        title for title in titles if re.search(rf'\b{re.escape(keyword)}\b', title, re.IGNORECASE)
    ]

# 保存
with open("keyword_to_titles.json", "w", encoding="utf-8") as f:
    json.dump(keyword_to_titles, f, ensure_ascii=False, indent=2)

# 4. 可視化
# 棒グラフ
plt.figure(figsize=(10, 6))
plt.bar(top_keywords.keys(), top_keywords.values())
plt.xticks(rotation=45)
plt.title('Top 20 Keywords in ACL 2025 Findings Papers')
plt.xlabel('Keywords')
plt.ylabel('TF-IDF Score')
plt.tight_layout()
plt.savefig('top_keywords_acl_2025.png')
plt.show()

# ワードクラウド
wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(top_keywords)
plt.figure(figsize=(15, 7.5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Keyword Cloud')
plt.show()

# 結果の確認