# %% [markdown]
# # Export Image Bytes (天级别增量, Spark 并发下载)
#
# 基于 export_content.py：前者写 `image` 列（单 URL 字符串），本脚本把 URL
# 下载并上传到 S3，写一份新 parquet `image_s3_path` 列附回，供 GPU encoding
# 直接读 S3 字节、不打外部 CDN。
#
# - 图片字节： s3://.../feed_content_images/<md5[:2]>/<md5>.jpg (flat, 跨日期去重)
# - 每日 parquet: S3_CONTENT_TEXT_EXPOSED_S3/<date>/  (原列 + image_s3_path)
#
# 幂等：已上传的 object 用 head_object 命中跳过，rerun 零成本。

# %% [markdown]
# ## Configuration

# %%
from datetime import datetime, timedelta

from config import S3_CONTENT_TEXT_EXPOSED

# 范围模式 (与 export_content.py 保持一致)
DATE_KEY_START = "2026-01-01"
DATE_KEY_END = "2026-03-31"

# 单日模式: DATE_KEY_START == DATE_KEY_END
# DATE_KEY_START = "2026-04-01"
# DATE_KEY_END = "2026-04-01"

_BASE = S3_CONTENT_TEXT_EXPOSED.rsplit("/", 1)[0]
S3_CONTENT_IMAGES        = f"{_BASE}/feed_content_images"
S3_CONTENT_TEXT_EXPOSED_S3 = f"{_BASE}/feed_content_text_exposed_s3"

HTTP_TIMEOUT = 5                   # 秒
MAX_BYTES = 5 * 1024 * 1024        # 5MB 单图上限
DOWNLOAD_PARTITIONS = 200          # Spark 下载并发度

print(f"Configuration:")
print(f"  Date range:   {DATE_KEY_START} ~ {DATE_KEY_END}")
print(f"  Input:        {S3_CONTENT_TEXT_EXPOSED}")
print(f"  Image bytes:  {S3_CONTENT_IMAGES}")
print(f"  Output:       {S3_CONTENT_TEXT_EXPOSED_S3}")


def date_range(start: str, end: str):
    d = datetime.strptime(start, "%Y-%m-%d")
    end_d = datetime.strptime(end, "%Y-%m-%d")
    while d <= end_d:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


# %% [markdown]
# ## Spark UDF (mapInPandas) — 并发下载 + 上传 S3

# %%
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType

DOWNLOAD_OUT_SCHEMA = StructType([
    StructField("image",         StringType(), False),
    StructField("image_s3_path", StringType(), True),
    StructField("status",        StringType(), False),
    StructField("size_bytes",    LongType(),   True),
])


def make_download_fn(s3_images_uri: str, timeout: int, max_bytes: int):
    """闭包工厂 — bind 常量进 UDF，避免 cloudpickle 在 executor 上拿不到 globals。"""

    def _download_partition(iterator):
        import hashlib
        from urllib.parse import urlparse

        import boto3
        import pandas as pd
        import requests
        from requests.adapters import HTTPAdapter

        parsed = urlparse(s3_images_uri)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")

        s3 = boto3.client("s3")
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        for pdf in iterator:
            rows = []
            for url in pdf["image"].tolist():
                if not url:
                    rows.append((url, None, "empty", None))
                    continue

                h = hashlib.md5(url.encode()).hexdigest()
                key = f"{prefix}/{h[:2]}/{h}.jpg"
                s3_path = f"s3://{bucket}/{key}"

                # 幂等: 已存在就跳过
                try:
                    head = s3.head_object(Bucket=bucket, Key=key)
                    rows.append((url, s3_path, "cached", int(head["ContentLength"])))
                    continue
                except Exception:
                    pass

                try:
                    resp = session.get(url, timeout=timeout, stream=True)
                    resp.raise_for_status()
                    body = resp.raw.read(max_bytes + 1)
                    if len(body) > max_bytes:
                        rows.append((url, None, "too_large", len(body)))
                        continue
                    if not body:
                        rows.append((url, None, "empty_body", 0))
                        continue
                    s3.put_object(Bucket=bucket, Key=key, Body=body)
                    rows.append((url, s3_path, "ok", len(body)))
                except requests.exceptions.Timeout:
                    rows.append((url, None, "timeout", None))
                except requests.exceptions.HTTPError as e:
                    code = e.response.status_code if e.response is not None else -1
                    rows.append((url, None, f"http_{code}", None))
                except Exception as e:
                    rows.append((url, None, f"error_{type(e).__name__}", None))

            yield pd.DataFrame(
                rows,
                columns=["image", "image_s3_path", "status", "size_bytes"],
            )

    return _download_partition


# %% [markdown]
# ## 逐天下载并附加 image_s3_path 列

# %%
for date_str in date_range(DATE_KEY_START, DATE_KEY_END):
    in_path = f"{S3_CONTENT_TEXT_EXPOSED}/{date_str}"
    out_path = f"{S3_CONTENT_TEXT_EXPOSED_S3}/{date_str}"

    print(f"\n{'='*60}")
    print(f"Processing {date_str}")
    print(f"  in:  {in_path}")
    print(f"  out: {out_path}")

    try:
        content = spark.read.parquet(in_path)
    except Exception as e:
        print(f"  SKIP: {e}")
        continue

    n_rows = content.count()
    n_with_img = content.filter(F.col("image").isNotNull() & (F.col("image") != "")).count()
    print(f"  {n_rows:,} rows, {n_with_img:,} with image URL")

    # Distinct URL dedup (同一天内同图只下一次；跨天由 S3 head_object 去重)
    distinct_urls = (
        content
        .select("image")
        .filter(F.col("image").isNotNull() & (F.col("image") != ""))
        .distinct()
    )
    n_distinct = distinct_urls.count()
    print(f"  {n_distinct:,} distinct URLs -> downloading across "
          f"{DOWNLOAD_PARTITIONS} partitions...")

    mapping = (
        distinct_urls
        .repartition(DOWNLOAD_PARTITIONS)
        .mapInPandas(
            make_download_fn(S3_CONTENT_IMAGES, HTTP_TIMEOUT, MAX_BYTES),
            schema=DOWNLOAD_OUT_SCHEMA,
        )
    )

    # 落盘 mapping（避免 join 时重跑 UDF）
    mapping_path = f"{out_path}/_url_mapping"
    mapping.write.mode("overwrite").option("compression", "snappy").parquet(mapping_path)

    mapping_reread = spark.read.parquet(mapping_path)
    print("  Status distribution:")
    mapping_reread.groupBy("status").count().orderBy(F.desc("count")) \
        .show(20, truncate=False)

    # Join s3_path 回 content
    result = content.join(
        mapping_reread.select("image", "image_s3_path"),
        on="image",
        how="left",
    )

    (
        result.repartition(24)
        .sortWithinPartitions("content_id")
        .write.mode("overwrite")
        .option("compression", "snappy")
        .parquet(out_path)
    )

    out_cnt = spark.read.parquet(out_path).count()
    print(f"  Wrote {out_cnt:,} rows -> {out_path}")


# %% [markdown]
# ## 验证

# %%
from pyspark.sql.functions import col

print(f"\n{'='*60}")
print("Verification Summary")
print(f"{'='*60}")

total_rows = 0
total_downloaded = 0
for date_str in date_range(DATE_KEY_START, DATE_KEY_END):
    path = f"{S3_CONTENT_TEXT_EXPOSED_S3}/{date_str}"
    try:
        df = spark.read.parquet(path)
        cnt = df.count()
        has_url = df.filter(col("image").isNotNull() & (col("image") != "")).count()
        has_s3 = df.filter(col("image_s3_path").isNotNull()).count()
        pct = 100 * has_s3 / max(has_url, 1)
        total_rows += cnt
        total_downloaded += has_s3
        print(f"  {date_str}: {cnt:,} rows | URL {has_url:,} | s3 {has_s3:,} ({pct:.1f}%)")
    except Exception as e:
        print(f"  {date_str}: ERROR - {e}")

print(f"\nTotal rows: {total_rows:,} | Total with s3: {total_downloaded:,}")
print(f"{'='*60}")
