# %% [markdown]
# # Export Exposed Content (文本 + 图片)
#
# 按日期范围一次性导出曝光内容，用于 embedding 编码。

# %% [markdown]
# ## Configuration

# %%
from gr_demo.config import (
    S3_CONTENT_TEXT_EXPOSED,
    HIVE_BEHAVIOR_TABLE, HIVE_CONTENTS_TABLE,
)

DATE_KEY_START = "2026-03-24"
DATE_KEY_END = "2026-03-31"
MAX_TEXT_LENGTH = 2048

date_key_start = int(DATE_KEY_START.replace("-", ""))
date_key_end = int(DATE_KEY_END.replace("-", ""))

CONTENT_OUTPUT_PATH = f"{S3_CONTENT_TEXT_EXPOSED}/{DATE_KEY_END}"

print(f"Configuration:")
print(f"  Date range: {DATE_KEY_START} ~ {DATE_KEY_END} (date_key {date_key_start} ~ {date_key_end})")
print(f"  Max text length: {MAX_TEXT_LENGTH}")
print(f"  Content output: {CONTENT_OUTPUT_PATH}")

# %% [markdown]
# ## 查询内容数据

# %%
content_query = f"""
WITH exposed_ids AS (
    SELECT DISTINCT df_5 AS content_id
    FROM {HIVE_BEHAVIOR_TABLE}
    WHERE date_key BETWEEN {date_key_start} AND {date_key_end}
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
WHERE 1=1
    AND p.db_create_time >= '{DATE_KEY_START}'
    AND p.db_create_time < '{DATE_KEY_END}'
    AND p.body_text_only IS NOT NULL
    AND LENGTH(p.body_text_only) > 10
"""

print(f"Content Query:\n{content_query}")
content_df = spark.sql(content_query)

# %% [markdown]
# ## Export Content to S3

# %%
from pyspark.sql.functions import col

content_df.repartition(24) \
    .sortWithinPartitions(col("create_date").desc()) \
    .write \
    .mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(CONTENT_OUTPUT_PATH)
print(f"Content exported to {CONTENT_OUTPUT_PATH}")

# %% [markdown]
# ## 验证内容导出结果

# %%
from pyspark.sql.functions import length, col

exported_content_df = spark.read.parquet(CONTENT_OUTPUT_PATH)
exported_content_df.show(5, truncate=80)
print(f"Total content exported: {exported_content_df.count():,}")

exported_content_df.select(
    length(col("full_text")).alias("text_len")
).summary("count", "mean", "min", "25%", "50%", "75%", "max").show()

print("\nImage null count:")
print(f"  NULL image: {exported_content_df.filter(col('image').isNull()).count():,}")
print(f"  Has image:  {exported_content_df.filter(col('image').isNotNull()).count():,}")
