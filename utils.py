# utils.py ────────────────────────────────────────────────────────────────
"""
Utility helpers for DatasetMerger

• Calls OpenAI gpt-3.5-turbo to align category names between two COCO datasets.
• If the API is unavailable it falls back to a quick difflib fuzzy-match.
• copy_and_rename_images now searches each entire extraction folder (recursively).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ───────────────────────────── logging ─────────────────────────────
log = logging.getLogger("dataset-merger")

# ─────────────────────── OpenAI configuration ─────────────────────
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")  # must be set in environment or .env
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TIMEOUT = 30  # seconds

# ──────────────── helper: fuzzy fallback matcher ────────────────
def _fuzzy_fallback(
    names_A: List[str], names_B: List[str], thresh: float = 0.85
) -> Dict[str, Optional[str]]:
    """
    If OpenAI fails or no key is provided, perform a simple fuzzy match:
    map each name in A to the single name in B whose lowercase similarity ≥ thresh.
    """
    mapping: Dict[str, Optional[str]] = {}
    for a in names_A:
        best_b, best_score = None, 0.0
        for b in names_B:
            score = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
            if score > best_score:
                best_b, best_score = b, score
        mapping[a] = best_b if best_score >= thresh else None
    return mapping


# ─────────────────────── main label-mapping fn ───────────────────────
def get_label_mapping(
    names_A: List[str], names_B: List[str]
) -> Dict[str, Optional[str]]:
    """
    Map category names in list A to names in list B using GPT-3.5-turbo.
    Falls back to fuzzy string match if:
      • OPENAI_API_KEY is not set
      • the API call fails
      • or the result is not valid JSON mapping every A-name
    """
    if not openai.api_key:
        log.warning("⚠️  OPENAI_API_KEY not set – using fuzzy fallback")
        return _fuzzy_fallback(names_A, names_B)

    prompt = (
        "Below are two lists of COCO category names.\n"
        "Return a JSON object whose keys are the names in list A and whose "
        "values are the BEST-MATCH name in list B or null if none.\n\n"
        f"List A:\n{names_A}\n\nList B:\n{names_B}\n"
    )

    try:
        log.info(" • GPT-3.5 label mapping …")
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
            timeout=OPENAI_TIMEOUT,      # ← use “timeout” instead of “request_timeout”
        )
        text = response.choices[0].message.content.strip()
        mapping = json.loads(text)
        # Ensure every A-name appears in the mapping (fill missing with None)
        for a in names_A:
            mapping.setdefault(a, None)
        return mapping
    except Exception as e:
        log.warning("⚠️  OpenAI failed (%s) – using fuzzy fallback", e)
        return _fuzzy_fallback(names_A, names_B)


# ───────────────────────── COCO helpers ──────────────────────────
def load_coco_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_categories(coco: Dict) -> Dict[int, Tuple[str, str]]:
    """
    Returns:
      { category_id: (category_name, supercategory) }
    """
    return {
        cat["id"]: (cat["name"], cat.get("supercategory", ""))
        for cat in coco.get("categories", [])
    }


def build_unified_categories(
    catsA: Dict[int, Tuple[str, str]],
    catsB: Dict[int, Tuple[str, str]],
    name_map_A_to_B: Dict[str, Optional[str]],
) -> Tuple[List[Dict], Dict[int, int], Dict[int, int]]:
    """
    Produces a merged 'categories' array for COCO and two remapping dicts:
      • remap_A: old_A_cat_id → new_unified_id
      • remap_B: old_B_cat_id → new_unified_id
    """
    unified: List[Dict] = []
    remap_A: Dict[int, int] = {}
    remap_B: Dict[int, int] = {}
    next_id = 1

    # Build lookup for B-name → B-id
    name_to_id_B = {v[0]: cid for cid, v in catsB.items()}

    # 1) Add categories from A, merging if name_map_A_to_B matches something in B
    for cidA, (nameA, superA) in catsA.items():
        matched = name_map_A_to_B.get(nameA)
        if matched and matched in name_to_id_B:
            cidB = name_to_id_B[matched]
            unified.append({"id": next_id, "name": nameA, "supercategory": superA})
            remap_A[cidA] = remap_B[cidB] = next_id
            next_id += 1
        else:
            unified.append({"id": next_id, "name": nameA, "supercategory": superA})
            remap_A[cidA] = next_id
            next_id += 1

    # 2) Add remaining B categories that weren’t merged
    for cidB, (nameB, superB) in catsB.items():
        if cidB not in remap_B:
            unified.append({"id": next_id, "name": nameB, "supercategory": superB})
            remap_B[cidB] = next_id
            next_id += 1

    return unified, remap_A, remap_B


def remap_images(
    imgsA: List[Dict], imgsB: List[Dict]
) -> Tuple[List[Dict], Dict[int, int], Dict[int, int]]:
    """
    Assign new image IDs so that A and B don’t collide.
    Returns:
      merged = [ { ..., "id": new_id } for each image in A then B ]
      mapA = { old_A_id: new_id }
      mapB = { old_B_id: new_id }
    """
    merged: List[Dict] = []
    mapA: Dict[int, int] = {}
    mapB: Dict[int, int] = {}
    nxt = 1

    for img in imgsA:
        mapA[img["id"]] = nxt
        merged.append({**img, "id": nxt})
        nxt += 1

    for img in imgsB:
        mapB[img["id"]] = nxt
        merged.append({**img, "id": nxt})
        nxt += 1

    return merged, mapA, mapB


def remap_annotations(
    annA: List[Dict],
    annB: List[Dict],
    mapA: Dict[int, int],
    mapB: Dict[int, int],
    cmapA: Dict[int, int],
    cmapB: Dict[int, int],
) -> List[Dict]:
    """
    Re-IDs and re-links each annotation so image_id and category_id refer
    to the newly merged sets. Returns a single merged list of all annotations.
    """
    merged: List[Dict] = []
    nxt = 1

    for a in annA:
        merged.append(
            {
                **a,
                "id": nxt,
                "image_id": mapA[a["image_id"]],
                "category_id": cmapA[a["category_id"]],
            }
        )
        nxt += 1

    for a in annB:
        merged.append(
            {
                **a,
                "id": nxt,
                "image_id": mapB[a["image_id"]],
                "category_id": cmapB[a["category_id"]],
            }
        )
        nxt += 1

    return merged


def copy_and_rename_images(
    extractA: Path, extractB: Path, merged_images: List[Dict], out_dir: Path
) -> None:
    """
    Recursively search within extractA and extractB for each image filename
    listed in 'merged_images'. Copies to 'out_dir', prefixing with "A_" or "B_".
    Raises FileNotFoundError if any filename is missing entirely.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a map: filename → absolute Path (for every file under extractA/extractB)
    all_A: Dict[str, Path] = {}
    all_B: Dict[str, Path] = {}

    if extractA.exists():
        for p in extractA.rglob("*"):
            if p.is_file():
                all_A[p.name] = p

    if extractB.exists():
        for p in extractB.rglob("*"):
            if p.is_file():
                all_B[p.name] = p

    # Now copy each merged image by matching only on the basename
    for img_info in merged_images:
        fname = img_info["file_name"]

        if fname in all_A:
            src_path = all_A[fname]
            prefix = "A_"
        elif fname in all_B:
            src_path = all_B[fname]
            prefix = "B_"
        else:
            # Neither A nor B contained this file anywhere
            raise FileNotFoundError(
                f"Could not find image '{fname}' under {extractA} or {extractB}"
            )

        dst_path = out_dir / f"{prefix}{fname}"
        shutil.copyfile(src_path, dst_path)


def save_coco_json(coco: Dict, path: Path) -> None:
    """
    Writes the merged COCO dictionary to the target 'path', creating
    parent folders as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
