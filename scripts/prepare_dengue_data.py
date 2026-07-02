from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen
import zipfile

import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dengue_pipeline.cleaner import (  # noqa: E402
    ANALYSIS_COLUMNS,
    REQUIRED_STANDARDIZED_COLUMNS,
    DengueDataCleaner,
    classification_counts,
)
from dengue_pipeline.features import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_COLUMNS,
)
from dengue_pipeline.paths import (  # noqa: E402
    DENGUE_YEARS,
    RAW_DOWNLOAD_DIR,
    analysis_dataset_path,
    ml_dataset_path,
)
from dengue_pipeline.sinan_mappings import COLUMN_RENAME_MAP  # noqa: E402


MANIFEST_PATH = PROJECT_ROOT / "data" / "dengue_manifest.json"
AUDIT_PATH = PROJECT_ROOT / "reports" / "data" / "dengue_data_audit.csv"


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_schema = {
        "raw_required_columns": required_raw_columns(),
        "analysis_columns": list(ANALYSIS_COLUMNS),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
    }
    if manifest.get("schema") != expected_schema:
        raise RuntimeError(
            "Manifest schema differs from the implemented pipeline. "
            "Update both explicitly before processing data."
        )
    for year in DENGUE_YEARS:
        checksum = manifest.get("years", {}).get(str(year), {}).get("sha256")
        if not checksum:
            raise RuntimeError(f"Manifest SHA-256 is missing for {year}")
    return manifest


def parse_years(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return DENGUE_YEARS
    years = tuple(sorted({int(value) for value in raw.split(",")}))
    unsupported = set(years) - set(DENGUE_YEARS)
    if unsupported:
        raise ValueError(f"Unsupported years: {sorted(unsupported)}")
    return years


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_resource(
    year: int,
    resource: dict,
    force: bool,
) -> tuple[Path, str]:
    RAW_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DOWNLOAD_DIR / f"DENGBR{str(year)[-2:]}.csv.zip"
    expected_hash = resource["sha256"].lower()
    expected_size = int(resource["size_bytes"])

    if destination.exists() and not force:
        actual_hash = sha256_file(destination)
        if destination.stat().st_size == expected_size and (
            not expected_hash or actual_hash == expected_hash
        ):
            return destination, actual_hash

    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    downloaded = 0
    print(f"[{year}] downloading {resource['url']}", flush=True)
    with urlopen(resource["url"], timeout=180) as response, temporary.open(
        "wb"
    ) as output:
        while chunk := response.read(4 * 1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            downloaded += len(chunk)

    actual_hash = digest.hexdigest()
    if downloaded != expected_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"[{year}] size mismatch: expected {expected_size}, "
            f"received {downloaded}"
        )
    if expected_hash and actual_hash != expected_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"[{year}] SHA-256 mismatch: expected {expected_hash}, "
            f"received {actual_hash}"
        )
    temporary.replace(destination)
    return destination, actual_hash


def required_raw_columns() -> list[str]:
    reverse = {
        standardized: raw
        for raw, standardized in COLUMN_RENAME_MAP.items()
    }
    missing = REQUIRED_STANDARDIZED_COLUMNS - set(reverse)
    if missing:
        raise RuntimeError(
            "Required standardized columns without SINAN mapping: "
            f"{sorted(missing)}"
        )
    return sorted(reverse[column] for column in REQUIRED_STANDARDIZED_COLUMNS)


class ParquetChunkWriter:
    def __init__(self, destination: Path):
        self.destination = destination
        self.temporary = destination.with_suffix(destination.suffix + ".part")
        self.writer: pq.ParquetWriter | None = None

    def write(self, frame: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            self.temporary.unlink(missing_ok=True)
            self.writer = pq.ParquetWriter(
                self.temporary,
                table.schema,
                compression="zstd",
            )
        else:
            table = pa.Table.from_pandas(
                frame,
                schema=self.writer.schema,
                preserve_index=False,
                safe=False,
            )
        self.writer.write_table(table, row_group_size=len(frame))

    def close(self) -> None:
        if self.writer is None:
            raise RuntimeError(f"No rows written to {self.destination}")
        self.writer.close()
        self.writer = None
        self.temporary.replace(self.destination)

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.temporary.unlink(missing_ok=True)


def csv_member(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]
    if len(members) != 1:
        raise RuntimeError(
            f"Expected one CSV in {zip_path}, found {members}"
        )
    return members[0]


def inspect_header(zip_path: Path, member: str) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as file:
        header = pd.read_csv(
            file,
            nrows=0,
            encoding="latin1",
        )
    return [str(column).strip().upper() for column in header.columns]


def prepare_year(
    year: int,
    resource: dict,
    zip_path: Path,
    actual_hash: str,
    chunk_size: int,
) -> dict:
    member = csv_member(zip_path)
    header = inspect_header(zip_path, member)
    if len(header) != int(resource["raw_columns"]):
        raise RuntimeError(
            f"[{year}] column count mismatch: expected "
            f"{resource['raw_columns']}, received {len(header)}"
        )

    selected_columns = required_raw_columns()
    missing_raw = set(selected_columns) - set(header)
    if missing_raw:
        raise RuntimeError(
            f"[{year}] required raw columns missing: {sorted(missing_raw)}"
        )

    analysis_writer = ParquetChunkWriter(analysis_dataset_path(year))
    ml_writer = ParquetChunkWriter(ml_dataset_path(year))
    raw_rows = 0
    accepted_rows = 0
    class_counter: Counter[str] = Counter()
    symptom_missing: Counter[str] = Counter()
    analysis_nulls: Counter[str] = Counter()
    process = psutil.Process()
    peak_rss_bytes = process.memory_info().rss

    try:
        with zipfile.ZipFile(zip_path) as archive, archive.open(member) as file:
            chunks = pd.read_csv(
                file,
                usecols=selected_columns,
                dtype="string",
                chunksize=chunk_size,
                encoding="latin1",
                low_memory=False,
            )
            for index, raw_chunk in enumerate(chunks, start=1):
                raw_chunk.columns = raw_chunk.columns.str.upper()
                raw_rows += len(raw_chunk)
                class_counter.update(
                    classification_counts(raw_chunk["CLASSI_FIN"])
                )

                analysis = DengueDataCleaner.transformar_analise_chunk(
                    raw_chunk,
                    year,
                )
                accepted_rows += len(analysis)
                analysis_nulls.update(
                    {
                        column: int(count)
                        for column, count in analysis.isna().sum().items()
                        if count
                    }
                )
                for symptom in (
                    "fever",
                    "myalgia",
                    "headache",
                    "rash",
                    "vomiting",
                    "nausea",
                    "back_pain",
                    "conjunctivitis",
                    "arthritis",
                    "joint_pain",
                    "petechiae",
                    "retro_orbital_pain",
                ):
                    symptom_missing[symptom] += int(
                        analysis[symptom].isna().sum()
                    )

                analysis_writer.write(analysis)
                ml_writer.write(
                    DengueDataCleaner.transformar_ml(analysis)
                )
                peak_rss_bytes = max(
                    peak_rss_bytes,
                    process.memory_info().rss,
                )
                print(
                    f"[{year}] chunk {index}: "
                    f"{raw_rows:,} raw / {accepted_rows:,} accepted",
                    flush=True,
                )

        expected_counts = {
            str(code): int(count)
            for code, count in resource["class_counts"].items()
        }
        actual_counts = dict(sorted(class_counter.items()))
        if raw_rows != int(resource["raw_rows"]):
            raise RuntimeError(
                f"[{year}] row count mismatch: expected "
                f"{resource['raw_rows']}, received {raw_rows}"
            )
        if actual_counts != dict(sorted(expected_counts.items())):
            raise RuntimeError(
                f"[{year}] CLASSI_FIN counts mismatch: "
                f"expected={expected_counts}, received={actual_counts}"
            )

        negative_rows = int(class_counter.get("5", 0))
        positive_codes = (
            ("1", "2", "3", "4", "10", "11", "12")
            if year <= 2016
            else ("10", "11", "12")
        )
        positive_rows = sum(class_counter.get(code, 0) for code in positive_codes)
        if not positive_rows or not negative_rows:
            raise RuntimeError(
                f"[{year}] both target classes are required: "
                f"positive={positive_rows}, negative={negative_rows}"
            )
        if accepted_rows != positive_rows + negative_rows:
            raise RuntimeError(
                f"[{year}] accepted count mismatch: expected "
                f"{positive_rows + negative_rows}, received {accepted_rows}"
            )

        analysis_writer.close()
        ml_writer.close()
    except Exception:
        analysis_writer.abort()
        ml_writer.abort()
        raise

    known_codes = {
        "",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
    }
    unexpected = {
        code: count
        for code, count in class_counter.items()
        if code not in known_codes
    }
    accepted_codes = {*positive_codes, "5"}
    removed_classifications = {
        code: count
        for code, count in sorted(class_counter.items())
        if code not in accepted_codes
    }
    return {
        "year": year,
        "source_sha256": actual_hash,
        "raw_rows": raw_rows,
        "accepted_rows": accepted_rows,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "removed_rows": raw_rows - accepted_rows,
        "peak_rss_gib": round(peak_rss_bytes / (1024**3), 3),
        "classification_counts": json.dumps(
            dict(sorted(class_counter.items())),
            sort_keys=True,
        ),
        "removed_classifications": json.dumps(
            removed_classifications,
            sort_keys=True,
        ),
        "unexpected_classifications": json.dumps(
            unexpected,
            sort_keys=True,
        ),
        "analysis_null_counts": json.dumps(
            dict(sorted(analysis_nulls.items())),
            sort_keys=True,
        ),
        "symptom_missing": json.dumps(
            dict(sorted(symptom_missing.items())),
            sort_keys=True,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare SINAN dengue data for 2014-2021."
    )
    parser.add_argument(
        "--years",
        help="Comma-separated years; defaults to 2014-2021.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
    )
    args = parser.parse_args()

    years = parse_years(args.years)
    if args.chunk_size < 10_000:
        parser.error("--chunk-size must be at least 10000")

    manifest = load_manifest()
    audit_rows = []
    for year in years:
        resource = manifest["years"][str(year)]
        zip_path, actual_hash = download_resource(
            year,
            resource,
            force=args.force_download,
        )
        if not resource["sha256"]:
            print(
                f"[{year}] WARNING: unpinned SHA-256 is {actual_hash}",
                flush=True,
            )
        if args.download_only:
            continue
        audit_rows.append(
            prepare_year(
                year,
                resource,
                zip_path,
                actual_hash,
                args.chunk_size,
            )
        )

    if audit_rows:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        audit = pd.DataFrame(audit_rows)
        if AUDIT_PATH.exists():
            previous = pd.read_csv(AUDIT_PATH)
            previous = previous[
                ~previous["year"].isin(audit["year"])
            ]
            audit = pd.concat([previous, audit], ignore_index=True)
        audit.sort_values("year").to_csv(
            AUDIT_PATH,
            index=False,
            encoding="utf-8",
        )
        print(f"Audit written to {AUDIT_PATH}")


if __name__ == "__main__":
    main()
