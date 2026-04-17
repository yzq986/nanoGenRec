# %% [markdown]
# # Export Exposed Content (天级别增量)
#
# 逐天导出曝光内容，每天一个独立 S3 目录，便于每日增量跑。

# %% [markdown]
# ## Configuration

# %%
from datetime import datetime, timedelta

from gr_demo.config import (
    S3_CONTENT_TEXT_EXPOSED,
    HIVE_BEHAVIOR_TABLE, HIVE_CONTENTS_TABLE,
)

# 范围模式 (补跑多天)
DATE_KEY_START = "2026-01-01"
DATE_KEY_END = "2026-03-31"

# 单日模式: 设置 DATE_KEY_START == DATE_KEY_END
# DATE_KEY_START = "2026-04-01"
# DATE_KEY_END = "2026-04-01"

MAX_TEXT_LENGTH = 2048

print(f"Configuration:")
print(f"  Date range: {DATE_KEY_START} ~ {DATE_KEY_END}")
print(f"  Max text length: {MAX_TEXT_LENGTH}")
print(f"  Content output base: {S3_CONTENT_TEXT_EXPOSED}")


def date_range(start: str, end: str):
    """Yield date strings from start to end (inclusive)."""
    d = datetime.strptime(start, "%Y-%m-%d")
    end_d = datetime.strptime(end, "%Y-%m-%d")
    while d <= end_d:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


# %% [markdown]
# ## 逐天导出内容数据

# %%
for date_str in date_range(DATE_KEY_START, DATE_KEY_END):
    date_key = int(date_str.replace("-", ""))  # e.g. 20260324

    content_query = f"""
WITH exposed_ids AS (
    SELECT DISTINCT df_5 AS content_id
    FROM {HIVE_BEHAVIOR_TABLE}
    WHERE date_key = {date_key}
      AND event_type = '$AppExposure'
      AND element_id IN (
          'app_exposure_view_home_feed_view','app_exposure_view_home_feed_view_v1','app_exposure_view_home_feed_view_v2',
          'app_exposure_view_home_feed_idle_view','app_exposure_view_home_feed_idle_view_v1','app_exposure_view_home_feed_idle_view_v2'
      )
      AND df_source IN ('discover','lite_discover','feed_discover','market_discover')
      AND df_5 IS NOT NULL AND df_5 != ''
)
SELECT
    p.id AS content_id,
    CAST(p.db_create_time AS DATE) AS create_date,
    SUBSTRING(
        IF(p.title IS NULL OR p.title = '',
           p.body_text_only,
           CONCAT(p.title, '\\n', p.body_text_only)
        ),
        1, {MAX_TEXT_LENGTH * 2}
    ) AS full_text,
    COALESCE(NULLIF(p.cover, ''), p.images, p.watermark_images) AS image
FROM {HIVE_CONTENTS_TABLE} p
INNER JOIN exposed_ids e ON p.id = e.content_id
"""

    print(f"\n{'='*60}")
    print(f"Exporting content for {date_str} (date_key={date_key})...")

    content_df = spark.sql(content_query)

    output_path = f"{S3_CONTENT_TEXT_EXPOSED}/{date_str}"
    content_df.repartition(24) \
        .sortWithinPartitions("content_id") \
        .write \
        .mode("overwrite") \
        .option("compression", "snappy") \
        .parquet(output_path)

    row_count = spark.read.parquet(output_path).count()
    print(f"Exported {date_str}: {row_count:,} rows -> {output_path}")

# %% [markdown]
# ## 验证导出结果

# %%
from pyspark.sql.functions import length, col

print(f"\n{'='*60}")
print("Verification Summary")
print(f"{'='*60}")

total_rows = 0
for date_str in date_range(DATE_KEY_START, DATE_KEY_END):
    path = f"{S3_CONTENT_TEXT_EXPOSED}/{date_str}"
    try:
        df = spark.read.parquet(path)
        cnt = df.count()
        total_rows += cnt
        null_img = df.filter(col('image').isNull()).count()
        print(f"  {date_str}: {cnt:,} rows, {null_img:,} null images")
    except Exception as e:
        print(f"  {date_str}: ERROR - {e}")

print(f"\nTotal rows across all days: {total_rows:,}")
print(f"{'='*60}")
