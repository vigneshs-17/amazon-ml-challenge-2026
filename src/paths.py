"""Central path definitions. Raw data is treated as read-only."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "student_resource"
DATASET = RAW_ROOT / "dataset"
TRAIN_DIR = DATASET / "train"
TEST_DIR = DATASET / "test"
REPORTS = PROJECT_ROOT / "reports"
INTERIM = PROJECT_ROOT / "data" / "interim"

SOURCE_FILES = {
    ("train", 1): TRAIN_DIR / "train_source1.tsv",
    ("train", 2): TRAIN_DIR / "train_source2.tsv",
    ("train", 3): TRAIN_DIR / "train_source3.tsv",
    ("test", 1): TEST_DIR / "test_source1.tsv",
    ("test", 2): TEST_DIR / "test_source2.tsv",
    ("test", 3): TEST_DIR / "test_source3.tsv",
}
GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"
