import requests
import pandas as pd
import time

API_KEY = "YX4CR0JFU23TR1JD"

print("=== 获取GOOG历史新闻 ===")

all_news = []

date_ranges = [
    ("20220101", "20220630"),
    ("20220701", "20221231"),
    ("20230101", "20230630"),
    ("20230701", "20231231"),
    ("20240101", "20240630"),
    ("20240701", "20241231"),
]

for time_from, time_to in date_ranges:
    url = (f"https://www.alphavantage.co/query"
           f"?function=NEWS_SENTIMENT"
           f"&tickers=GOOG"
           f"&time_from={time_from}T0000"
           f"&time_to={time_to}T2359"
           f"&limit=200"
           f"&apikey={API_KEY}")

    try:
        response = requests.get(url, timeout=30)
        data     = response.json()

        if 'feed' in data:
            for news in data['feed']:
                all_news.append({
                    'date':      news['time_published'][:8],
                    'title':     news['title'],
                    'sentiment': news.get('overall_sentiment_label', 'Neutral'),
                    'score':     float(news.get('overall_sentiment_score', 0))
                })
            print(f"{time_from[:6]}：获取到 {len(data['feed'])} 条新闻")
        else:
            print(f"{time_from[:6]}：{data}")

    except Exception as e:
        print(f"{time_from[:6]}：连接失败 - {e}")
        continue

    time.sleep(15)

df_news = pd.DataFrame(all_news)
df_news['date'] = pd.to_datetime(df_news['date'], format='%Y%m%d')
df_news = df_news.sort_values('date').reset_index(drop=True)

print(f"\n总共获取：{len(df_news)}条新闻")
print(f"时间范围：{df_news['date'].min()} 到 {df_news['date'].max()}")
print(f"\n情感分布：")
print(df_news['sentiment'].value_counts())

df_news.to_csv(
    '/Users/fengyihao/PyCharmMiscProject/深度学习/data/goog_real_news.csv',
    index=False
)
print("\n已保存到 goog_real_news.csv！")