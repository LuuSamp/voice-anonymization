import argparse
import csv
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SPEECH_MID = "/m/09x0r"
DEFAULT_PER_LABEL = 5
DEFAULT_WORKERS = 4
DEFAULT_OUTPUT = Path("datasets/audioset/audio")
CLASS_LABELS_PATH = Path("datasets/audioset/class_labels_indices.csv")
SEGMENTS_PATH = Path("datasets/audioset/balanced_train_segments.csv")
# Human-sound / speech branch in AudioSet ontology (indices 0–70, before Animal).
SPEECH_ONTOLOGY_END_INDEX = 70


def sanitize_folder_name(name: str) -> str:
    name = name.strip().strip('"')
    name = re.sub(r'[<>:"/\\|?*]', "-", name)
    return name or "unknown"


def download_clip(
    video_id: str,
    start_time: float,
    end_time: float,
    output_folder: Path,
    stdout: Path | None = None,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_path = output_folder / f"{video_id}_{start_time:.1f}s.wav"
    command = (
        f'yt-dlp -x --audio-format wav "{url}" '
        f'--postprocessor-args "-ss {start_time} -to {end_time}" '
        f'-o "{output_path}"'
    )
    if stdout is not None:
        command += f" > {stdout} 2>&1"
    else:
        command += " > /dev/null 2>&1"
    os.system(command)


def load_class_labels(path: Path) -> tuple[dict[str, str], dict[str, str], set[str]]:
    display_to_mid: dict[str, str] = {}
    mid_to_display: dict[str, str] = {}
    speech_mids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            mid = row["mid"].strip()
            display = row["display_name"].strip().strip('"')
            display_to_mid[display] = mid
            mid_to_display[mid] = display
            if index <= SPEECH_ONTOLOGY_END_INDEX:
                speech_mids.add(mid)
    return display_to_mid, mid_to_display, speech_mids


def normalize_labels(raw_labels: list[str]) -> list[str]:
    return [label.strip().strip('"') for label in raw_labels if label.strip()]


def load_segments(path: Path) -> list[tuple[str, float, float, list[str]]]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(
            filter(lambda row: row[0] != "#", f),
            delimiter=",",
            skipinitialspace=True,
        )
        return [
            (
                row["YTID"],
                float(row["start_seconds"]),
                float(row["end_seconds"]),
                normalize_labels(row["positive_labels"].split(",")),
            )
            for row in reader
        ]


def folder_name_for_mid(mid: str, mid_to_display: dict[str, str]) -> str:
    return sanitize_folder_name(mid_to_display.get(mid, mid))


def categorize_segment(
    labels: list[str],
    speech_mids: set[str],
    mid_to_display: dict[str, str],
    *,
    all_co_labels: bool,
) -> str | None:
    """Assign a segment to a download folder.

  Segments must contain the Speech label (``/m/09x0r``).

  - Solo speech: only the Speech label is present.
  - Multi-label: use the first label after Speech in the CSV list. When that label
    is not in the speech ontology and ``all_co_labels`` is false, use the first
    speech-ontology label among the remaining labels instead.
  """
    if SPEECH_MID not in labels:
        return None

    others = [label for label in labels if label != SPEECH_MID]
    if not others:
        return folder_name_for_mid(SPEECH_MID, mid_to_display)

    # First label other than Speech (AudioSet column order).
    key = others[0]
    if key not in speech_mids:
        if all_co_labels:
            return folder_name_for_mid(key, mid_to_display)
        speech_others = [label for label in others if label in speech_mids]
        if not speech_others:
            return None
        key = speech_others[0]

    return folder_name_for_mid(key, mid_to_display)


def segment_key(segment: tuple[str, float, float, list[str]]) -> tuple[str, float, float]:
    return segment[0], segment[1], segment[2]


def select_segments_per_label(
    segments: list[tuple[str, float, float, list[str]]],
    speech_mids: set[str],
    mid_to_display: dict[str, str],
    per_label: int,
    *,
    all_co_labels: bool,
) -> dict[str, list[tuple[str, float, float, list[str]]]]:
    buckets: dict[str, list[tuple[str, float, float, list[str]]]] = defaultdict(list)
    used: set[tuple[str, float, float]] = set()
    speech_folder = folder_name_for_mid(SPEECH_MID, mid_to_display)

    for segment in segments:
        labels = segment[3]
        if SPEECH_MID not in labels:
            continue
        others = [label for label in labels if label != SPEECH_MID]
        if others:
            continue
        key = segment_key(segment)
        if key in used or len(buckets[speech_folder]) >= per_label:
            continue
        buckets[speech_folder].append(segment)
        used.add(key)

    for segment in segments:
        key = segment_key(segment)
        if key in used:
            continue
        folder = categorize_segment(
            segment[3],
            speech_mids,
            mid_to_display,
            all_co_labels=all_co_labels,
        )
        if folder is None or folder == speech_folder:
            continue
        if len(buckets[folder]) >= per_label:
            continue
        buckets[folder].append(segment)
        used.add(key)

    if len(buckets[speech_folder]) < per_label:
        _fill_solo_speech_fallback(
            segments,
            buckets,
            speech_mids,
            speech_folder,
            per_label,
            used,
        )

    return dict(buckets)


def _fill_solo_speech_fallback(
    segments: list[tuple[str, float, float, list[str]]],
    buckets: dict[str, list[tuple[str, float, float, list[str]]]],
    speech_mids: set[str],
    speech_folder: str,
    per_label: int,
    used: set[tuple[str, float, float]],
) -> None:
    """Fill the Speech folder from speech-only clips when strict solo clips are absent."""
    needed = per_label - len(buckets[speech_folder])
    if needed <= 0:
        return

    candidates: list[tuple[int, tuple[str, float, float, list[str]]]] = []
    for segment in segments:
        key = segment_key(segment)
        if key in used:
            continue
        labels = segment[3]
        if SPEECH_MID not in labels or not set(labels).issubset(speech_mids):
            continue
        candidates.append((len(labels), segment))

    candidates.sort(key=lambda item: item[0])
    for _, segment in candidates[:needed]:
        buckets[speech_folder].append(segment)
        used.add(segment_key(segment))


def _build_download_jobs(
    buckets: dict[str, list[tuple[str, float, float, list[str]]]],
    output_root: Path,
) -> list[tuple[str, float, float, Path]]:
    jobs: list[tuple[str, float, float, Path]] = []
    for folder_name, segments in sorted(buckets.items()):
        label_dir = output_root / folder_name
        for video_id, start_time, end_time, _labels in segments:
            jobs.append((video_id, start_time, end_time, label_dir))
    return jobs


def _run_download_job(
    job: tuple[str, float, float, Path],
    stdout: Path | None,
) -> tuple[str, float, float]:
    video_id, start_time, end_time, label_dir = job
    download_clip(video_id, start_time, end_time, label_dir, stdout=stdout)
    return video_id, start_time, end_time


def download_segments(
    buckets: dict[str, list[tuple[str, float, float, list[str]]]],
    output_root: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    stdout: Path | None = None,
) -> None:
    jobs = _build_download_jobs(buckets, output_root)
    total = len(jobs)
    workers = max(1, workers)
    print(
        f"Downloading {total} segments across {len(buckets)} labels to {output_root} "
        f"({workers} worker{'s' if workers != 1 else ''})"
    )
    if workers > 1 and stdout is not None:
        print("Warning: --stdout log file is not used when workers > 1")

    try:
        from tqdm import tqdm

        progress = tqdm(total=total, desc="clips")
    except ImportError:
        progress = None

    n_done = 0
    if workers == 1:
        for job in jobs:
            _run_download_job(job, stdout)
            n_done += 1
            if progress is None:
                print(f"Downloaded clip {n_done}/{total}")
            else:
                progress.update(1)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_run_download_job, job, None)
                for job in jobs
            ]
            for future in as_completed(futures):
                future.result()
                n_done += 1
                if progress is not None:
                    progress.update(1)
                elif n_done % max(1, total // 10) == 0 or n_done == total:
                    print(f"Downloaded clip {n_done}/{total}")

    if progress is not None:
        progress.close()

    print(f"Downloaded {n_done} segments to {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download AudioSet clips that contain Speech, grouped into one folder "
            "per sub-label (default: speech-ontology labels only)."
        )
    )
    parser.add_argument(
        "-n",
        "--per-label",
        type=int,
        default=DEFAULT_PER_LABEL,
        help=f"Clips to download per label folder (default: {DEFAULT_PER_LABEL})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--class-labels",
        type=Path,
        default=CLASS_LABELS_PATH,
        help="Path to class_labels_indices.csv",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=SEGMENTS_PATH,
        help="Path to balanced_train_segments.csv",
    )
    parser.add_argument(
        "--all-co-labels",
        action="store_true",
        help="Also create folders for non-speech co-labels (e.g. Music, Dog)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print planned downloads without calling yt-dlp",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            f"Parallel download workers via thread pool (default: {DEFAULT_WORKERS}). "
            "Use 1 for sequential downloads."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, mid_to_display, speech_mids = load_class_labels(args.class_labels)
    segments = load_segments(args.segments)

    speech_segments = [segment for segment in segments if SPEECH_MID in segment[3]]
    buckets = select_segments_per_label(
        speech_segments,
        speech_mids,
        mid_to_display,
        args.per_label,
        all_co_labels=args.all_co_labels,
    )

    speech_folder = folder_name_for_mid(SPEECH_MID, mid_to_display)
    print(f"Speech segments in CSV: {len(speech_segments)}")
    print(f"Label folders selected: {len(buckets)}")
    print(f"'{speech_folder}' clips: {len(buckets.get(speech_folder, []))}")
    if len(buckets.get(speech_folder, [])) < args.per_label:
        print(
            "Note: balanced train has no segments with only the Speech label; "
            "the Speech folder is filled from the shortest speech-only annotations."
        )

    for folder_name in sorted(buckets):
        print(f"  {folder_name}: {len(buckets[folder_name])}")

    if args.list_only:
        return

    download_segments(buckets, args.output, workers=args.workers)


if __name__ == "__main__":
    main()
