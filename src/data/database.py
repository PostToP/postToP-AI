import logging
import psycopg2
import json
from dotenv import load_dotenv
import os
load_dotenv()

logger = logging.getLogger("experiment")

def get_connection():
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT")
    )
    return conn

def fetch_videos():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""SELECT
    title,
    description,
    name as channel_name,
    duration,
    language,
    main_category_id,
    cats,
    is_music
FROM
    posttop.video v
    INNER JOIN posttop.video_metadata vm ON v.id = vm.video_id
    INNER JOIN posttop.is_music_video imv ON v.id = imv.video_id
    INNER JOIN posttop.channel c ON v.channel_id = c.id
    LEFT JOIN (
        SELECT v.id, array_agg(cat.name) cats
        from posttop.video v
            INNER join posttop.video_category vc ON v.id = vc.video_id
            INNER JOIN posttop.category cat ON vc.category_id = cat.id
        GROUP BY
            v.id
    ) asd ON v.id = asd.id;""")
    videos = cursor.fetchall()
    cursor.close()
    conn.close()
    return videos

def convert_postgres_videos_to_json(videos):
    all_vids = []
    for video in videos:
        v = {}
        v["Title"] = video[0]
        v["Description"] = video[1]
        v["Channel Name"] = video[2]
        v["Duration"] = video[3]
        v["Language"] = video[4]
        v["Category"] = video[5]
        v["Categories"] = video[6] if video[6] else []
        v["Is Music"] = video[7]
        all_vids.append(v)
    return all_vids

def save_videos_to_json(videos, filename='dataset/videos.json'):
    with open(filename, 'w') as f:
        json.dump(videos, f, indent=4)

def main():
    videos = fetch_videos()
    if not videos:
        logger.error("No videos fetched from database")
        return

    video_json = convert_postgres_videos_to_json(videos)
    save_videos_to_json(video_json)
    logger.debug(f"Saved {len(video_json)} videos to dataset/videos.json")

if __name__ == "__main__":
    main()