import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#jsonlファイルを読み込み、各行を辞書として処理
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data 

with open('top_keywords_acl_2025.json', 'r', encoding='utf-8') as f:
    top_keywords = json.load(f)
#キーワードのリストを作成
keywords = list(top_keywords.keys())

data_list = read_jsonl('outputs.jsonl')

keywords_list = []
#キーワードごとにデータを抽出
for i in range(len(data_list)):
    ith_data = data_list[i]
    try:
        if ith_data['response']['body']['choices'][0]['message']['content'] == 'Yes':
            keywords_list.append(keywords[i])
    except KeyError:
        print(f"KeyError at index {i}: {ith_data}")
        continue

#top_keywordsのうち、keywords_listに含まれているものだけを抽出
filtered_keywords = {key: top_keywords[key] for key in keywords_list if key in top_keywords}
#保存
with open('filtered_keywords_acl_2025.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_keywords, f, indent=4)

# 4. 可視化
# 棒グラフ
plt.figure(figsize=(10, 6))
# 上位20キーワードを抽出
filtered_keywords = dict(sorted(filtered_keywords.items(), key=lambda x: x[1], reverse=True)[10:50])
plt.bar(filtered_keywords.keys(), filtered_keywords.values(), color='skyblue')
plt.xticks(rotation=45)
plt.title('Top 20 Keywords in ACL 2025 Findings Papers')
plt.xlabel('Keywords')
plt.ylabel('TF-IDF Score')
plt.tight_layout()
plt.savefig('top_keywords_acl_2025.png')
plt.show()

# ワードクラウド

wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(filtered_keywords)
plt.figure(figsize=(15, 7.5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Keyword Cloud')
plt.show()


        
#各要素をまとめて1つの辞書にする
print(keywords_list)
