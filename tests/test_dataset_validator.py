"""Unit tests for the Gray Cast Iron Dataset Validator.

Tests all automated integrity checks:
1. File pairs: orphaned images and orphaned masks.
2. Dimensions: image and mask dimension mismatches.
3. Mask values: non-binary pixels (anti-aliasing, soft edges).
4. Empty and inverted masks: 0% foreground and >40% foreground.
5. Duplicate images: exact hash duplication detection.
6. Split leakage: specimen-level cross-contamination between train and test.
7. Valid clean dataset: properly partitioned specimen-level dataset passing validation.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from services.gray_cast_iron.validate_dataset import (
    validate_split_directory,
    validate_gray_cast_iron_dataset,
    extract_specimen_id,
)


@pytest.fixture
def temp_dataset_dir():
    d = Path(tempfile.mkdtemp(prefix="test_gci_dataset_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _create_synthetic_micrograph(w: int = 256, h: int = 256, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    # Metallic background
    img = np.ones((h, w, 3), dtype=np.uint8) * 190
    # Add dark graphite flakes
    for _ in range(15):
        pt1 = (np.random.randint(20, w - 20), np.random.randint(20, h - 20))
        pt2 = (pt1[0] + np.random.randint(-15, 15), pt1[1] + np.random.randint(-15, 15))
        cv2.line(img, pt1, pt2, (40, 40, 40), thickness=2)
    return img


def _create_synthetic_mask(w: int = 256, h: int = 256, fg_ratio: float = 0.10) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    num_fg = int(w * h * fg_ratio)
    coords_y = np.random.randint(0, h, num_fg)
    coords_x = np.random.randint(0, w, num_fg)
    mask[coords_y, coords_x] = 255
    return mask


def test_extract_specimen_id():
    assert extract_specimen_id("specimen01_field02.png") == "specimen01"
    assert extract_specimen_id("heat_A_roi3.tif") == "heat_A"
    assert extract_specimen_id("sample05.jpg") == "sample05"
    assert extract_specimen_id("GCI-400-01-05.png") == "GCI-400-01"


def test_detects_orphaned_files(temp_dataset_dir):
    img_dir = temp_dataset_dir / "images"
    mask_dir = temp_dataset_dir / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    # Valid pair
    cv2.imwrite(str(img_dir / "spec01_f01.png"), _create_synthetic_micrograph())
    cv2.imwrite(str(mask_dir / "spec01_f01.png"), _create_synthetic_mask())

    # Orphaned image
    cv2.imwrite(str(img_dir / "spec01_f02.png"), _create_synthetic_micrograph())
    # Orphaned mask
    cv2.imwrite(str(mask_dir / "spec01_f03.png"), _create_synthetic_mask())

    res = validate_split_directory(img_dir, mask_dir, split_name="train")
    assert res["paired_count"] == 1
    assert "spec01_f02.png" in res["orphaned_images"]
    assert "spec01_f03.png" in res["orphaned_masks"]
    assert any("orphaned" in err.lower() or "without" in err.lower() for err in res["errors"])


def test_detects_dimension_mismatch(temp_dataset_dir):
    img_dir = temp_dataset_dir / "images"
    mask_dir = temp_dataset_dir / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    cv2.imwrite(str(img_dir / "spec01_f01.png"), _create_synthetic_micrograph(w=512, h=512))
    cv2.imwrite(str(mask_dir / "spec01_f01.png"), _create_synthetic_mask(w=256, h=256))

    res = validate_split_directory(img_dir, mask_dir, split_name="train")
    assert len(res["dimension_mismatches"]) == 1
    assert any("dimension mismatch" in err.lower() for err in res["errors"])


def test_detects_non_binary_mask_values(temp_dataset_dir):
    img_dir = temp_dataset_dir / "images"
    mask_dir = temp_dataset_dir / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    img = _create_synthetic_micrograph()
    # Create non-binary mask with smooth gradient values
    bad_mask = np.full((256, 256), 128, dtype=np.uint8)

    cv2.imwrite(str(img_dir / "spec01_f01.png"), img)
    cv2.imwrite(str(mask_dir / "spec01_f01.png"), bad_mask)

    res = validate_split_directory(img_dir, mask_dir, split_name="train")
    assert len(res["non_binary_masks"]) == 1
    assert any("non-binary" in err.lower() for err in res["errors"])


def test_detects_empty_and_inverted_masks(temp_dataset_dir):
    img_dir = temp_dataset_dir / "images"
    mask_dir = temp_dataset_dir / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    # Empty mask
    empty_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "empty_sample.png"), _create_synthetic_micrograph(seed=1))
    cv2.imwrite(str(mask_dir / "empty_sample.png"), empty_mask)

    # Inverted mask (85% foreground)
    inverted_mask = np.full((256, 256), 255, dtype=np.uint8)
    inverted_mask[:30, :30] = 0
    cv2.imwrite(str(img_dir / "inverted_sample.png"), _create_synthetic_micrograph(seed=2))
    cv2.imwrite(str(mask_dir / "inverted_sample.png"), inverted_mask)

    res = validate_split_directory(img_dir, mask_dir, split_name="train")
    assert len(res["empty_masks"]) == 1
    assert len(res["inverted_masks"]) == 1


def test_detects_duplicate_images(temp_dataset_dir):
    img_dir = temp_dataset_dir / "images"
    mask_dir = temp_dataset_dir / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    img = _create_synthetic_micrograph(seed=99)
    mask = _create_synthetic_mask()

    # Two files with identical image contents
    cv2.imwrite(str(img_dir / "sample_A.png"), img)
    cv2.imwrite(str(mask_dir / "sample_A.png"), mask)
    cv2.imwrite(str(img_dir / "sample_B.png"), img)
    cv2.imwrite(str(mask_dir / "sample_B.png"), mask)

    res = validate_split_directory(img_dir, mask_dir, split_name="train")
    assert len(res["duplicate_images"]) == 1
    assert any("duplicate image" in err.lower() for err in res["errors"])


def test_detects_specimen_level_split_leakage(temp_dataset_dir):
    # Setup partitioned dataset
    for split in ("train", "val", "test"):
        (temp_dataset_dir / split / "images").mkdir(parents=True)
        (temp_dataset_dir / split / "masks").mkdir(parents=True)

    # Specimen 01 in train
    cv2.imwrite(str(temp_dataset_dir / "train" / "images" / "spec01_f01.png"), _create_synthetic_micrograph(seed=1))
    cv2.imwrite(str(temp_dataset_dir / "train" / "masks" / "spec01_f01.png"), _create_synthetic_mask())

    # Specimen 01 also in val (LEAKAGE!)
    cv2.imwrite(str(temp_dataset_dir / "val" / "images" / "spec01_f02.png"), _create_synthetic_micrograph(seed=2))
    cv2.imwrite(str(temp_dataset_dir / "val" / "masks" / "spec01_f02.png"), _create_synthetic_mask())

    # Specimen 02 in test
    cv2.imwrite(str(temp_dataset_dir / "test" / "images" / "spec02_f01.png"), _create_synthetic_micrograph(seed=3))
    cv2.imwrite(str(temp_dataset_dir / "test" / "masks" / "spec02_f01.png"), _create_synthetic_mask())

    report = validate_gray_cast_iron_dataset(temp_dataset_dir)
    assert not report["is_valid"]
    assert len(report["leakage_errors"]) > 0
    assert any("spec01" in err for err in report["leakage_errors"])


def test_valid_partitioned_dataset_passes(temp_dataset_dir):
    # Proper partitioned dataset without leakage
    for split in ("train", "val", "test"):
        (temp_dataset_dir / split / "images").mkdir(parents=True)
        (temp_dataset_dir / split / "masks").mkdir(parents=True)

    # Train: specimens 01, 02
    for spec in ("spec01", "spec02"):
        cv2.imwrite(str(temp_dataset_dir / "train" / "images" / f"{spec}_f01.png"), _create_synthetic_micrograph(seed=hash(spec) % 10000))
        cv2.imwrite(str(temp_dataset_dir / "train" / "masks" / f"{spec}_f01.png"), _create_synthetic_mask())

    # Val: specimen 03
    cv2.imwrite(str(temp_dataset_dir / "val" / "images" / "spec03_f01.png"), _create_synthetic_micrograph(seed=303))
    cv2.imwrite(str(temp_dataset_dir / "val" / "masks" / "spec03_f01.png"), _create_synthetic_mask())

    # Test: specimen 04
    cv2.imwrite(str(temp_dataset_dir / "test" / "images" / "spec04_f01.png"), _create_synthetic_micrograph(seed=404))
    cv2.imwrite(str(temp_dataset_dir / "test" / "masks" / "spec04_f01.png"), _create_synthetic_mask())

    report = validate_gray_cast_iron_dataset(temp_dataset_dir)
    assert report["is_valid"] is True
    assert report["summary"]["total_paired"] == 4
    assert len(report["leakage_errors"]) == 0
    assert report["summary"]["error_count"] == 0


def test_automated_prepare_splits_prevents_leakage(temp_dataset_dir):
    from services.gray_cast_iron.prepare_splits import create_specimen_level_split

    raw_dir = temp_dataset_dir / "raw"
    target_dir = temp_dataset_dir / "partitioned"
    (raw_dir / "images").mkdir(parents=True)
    (raw_dir / "masks").mkdir(parents=True)

    # Create 5 specimens with 2 fields each (10 micrographs total)
    for i in range(1, 6):
        spec_id = f"heat{i:02d}"
        for f in range(1, 3):
            fname = f"{spec_id}_field{f:02d}.png"
            cv2.imwrite(str(raw_dir / "images" / fname), _create_synthetic_micrograph(seed=i * 10 + f))
            cv2.imwrite(str(raw_dir / "masks" / fname), _create_synthetic_mask())

    res = create_specimen_level_split(
        source_dir=raw_dir,
        output_dir=target_dir,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        seed=123,
    )

    assert res["is_valid"] is True
    assert res["transferred"]["train"] > 0
    assert res["transferred"]["val"] > 0
    assert res["transferred"]["test"] > 0
    # Confirm no split leakage in partitioned dataset
    val_check = validate_gray_cast_iron_dataset(target_dir)
    assert val_check["is_valid"] is True
    assert len(val_check["leakage_errors"]) == 0
