# utils.py

from llama_cpp import Llama
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

LOCAL_LLAMA_PATH = "/Users/yusufmorsy/llama-ggml/llama-2-7b-q8_0.ggufv3.bin"

llm = Llama(
    model_path=LOCAL_LLAMA_PATH,
    n_ctx=2048,
    verbose=False,
)


# ──────────────────────────────────────────────────────────────────────────────
# 2) Load a COCO JSON from disk
# ──────────────────────────────────────────────────────────────────────────────
def load_coco_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# 3) Extract categories from a loaded COCO JSON.
#    Returns a mapping { category_id: (category_name, supercategory) }.
# ──────────────────────────────────────────────────────────────────────────────
def extract_categories(coco: Dict) -> Dict[int, Tuple[str, str]]:
    cats: Dict[int, Tuple[str, str]] = {}
    for cat in coco.get("categories", []):
        cats[cat["id"]] = (cat["name"], cat.get("supercategory", ""))
    return cats


# ──────────────────────────────────────────────────────────────────────────────
# 4) Use LLaMA 2 to “match” category names from A → B.
#    names_A and names_B are each List[str].
#    Return Dict[name_in_A, matching_name_in_B_or_None].
# ──────────────────────────────────────────────────────────────────────────────
def get_label_mapping(names_A: List[str], names_B: List[str]) -> Dict[str, Optional[str]]:
    prompt_text = f"""
Below are two lists of COCO category names for Dataset A and Dataset B.
Your job is to tell me which names match. If a name in A corresponds to a name in B, return the B-name. If no reliable match, return null.

Dataset A names:
{json.dumps(names_A, indent=2)}

Dataset B names:
{json.dumps(names_B, indent=2)}

Please output a JSON dict mapping each name in A (as key) to the matching name in B (or null). For example:
{{ "cat": "feline", "dog": null, ... }}
""".strip()

    # Call llama_cpp
    response = llm.create_completion(
        prompt=prompt_text,
        max_tokens=512,
        temperature=0.0,
    )
    text = response["choices"][0]["text"].strip()
    return json.loads(text)


# ──────────────────────────────────────────────────────────────────────────────
# 5) Build unified categories list and remapping tables.
#
#    - catsA, catsB: Dict[int, (name, supercategory)] from extract_categories().
#    - name_map_A_to_B: Dict[name_in_A, matched_name_in_B_or_None] from step 4.
#
#    Return: (
#       unified_categories: List[{"id": int, "name": str, "supercategory": str}],
#       remap_A: Dict[old_A_cat_id, new_unified_id],
#       remap_B: Dict[old_B_cat_id, new_unified_id]
#    )
# ──────────────────────────────────────────────────────────────────────────────
def build_unified_categories(
    catsA: Dict[int, Tuple[str, str]],
    catsB: Dict[int, Tuple[str, str]],
    name_map_A_to_B: Dict[str, Optional[str]],
) -> Tuple[List[Dict], Dict[int, int], Dict[int, int]]:
    unified: List[Dict] = []
    remap_A: Dict[int, int] = {}
    remap_B: Dict[int, int] = {}
    next_id = 1

    # 5a) Add categories from A, merging if name_map_A_to_B says so
    for cidA, (nameA, superA) in catsA.items():
        matched = name_map_A_to_B.get(nameA)
        if matched is not None and matched in {v[0] for v in catsB.values()}:
            # Find B’s category ID
            for cidB, (nameB, superB) in catsB.items():
                if nameB == matched:
                    unified.append({"id": next_id, "name": nameA, "supercategory": superA})
                    remap_A[cidA] = next_id
                    remap_B[cidB] = next_id
                    next_id += 1
                    break
        else:
            # No match → add A’s category as-is
            unified.append({"id": next_id, "name": nameA, "supercategory": superA})
            remap_A[cidA] = next_id
            next_id += 1

    # 5b) Add any B categories that weren’t merged above
    for cidB, (nameB, superB) in catsB.items():
        if cidB not in remap_B:
            unified.append({"id": next_id, "name": nameB, "supercategory": superB})
            remap_B[cidB] = next_id
            next_id += 1

    return unified, remap_A, remap_B


# ──────────────────────────────────────────────────────────────────────────────
# 6) Remap image IDs so A and B don’t collide:
#
#    - cocoA_images, cocoB_images: List[{"id": int, "file_name": str, …}]
#    Returns:
#      merged_images: List[{"id": int, "file_name": str, …}]  # new IDs
#      img_id_map_A: Dict[old_A_id, new_id]
#      img_id_map_B: Dict[old_B_id, new_id]
# ──────────────────────────────────────────────────────────────────────────────
def remap_images(
    cocoA_images: List[Dict],
    cocoB_images: List[Dict],
) -> Tuple[List[Dict], Dict[int, int], Dict[int, int]]:
    merged: List[Dict] = []
    img_id_map_A: Dict[int, int] = {}
    img_id_map_B: Dict[int, int] = {}
    next_id = 1

    for img in cocoA_images:
        img_id_map_A[img["id"]] = next_id
        new_img = img.copy()
        new_img["id"] = next_id
        merged.append(new_img)
        next_id += 1

    for img in cocoB_images:
        img_id_map_B[img["id"]] = next_id
        new_img = img.copy()
        new_img["id"] = next_id
        merged.append(new_img)
        next_id += 1

    return merged, img_id_map_A, img_id_map_B


# ──────────────────────────────────────────────────────────────────────────────
# 7) Remap and merge annotations:
#
#    - cocoA_annos, cocoB_annos: List[{"id": int,"image_id": int,"category_id": int,…}]
#    - img_map_A, img_map_B: Dict[old_image_id, new_image_id] from step 6
#    - cat_map_A, cat_map_B: Dict[old_cat_id, new_cat_id] from step 5
#
#    Returns: merged_annotations: List[{"id": int,"image_id": int,"category_id": int,…}]
# ──────────────────────────────────────────────────────────────────────────────
def remap_annotations(
    cocoA_annos: List[Dict],
    cocoB_annos: List[Dict],
    img_map_A: Dict[int, int],
    img_map_B: Dict[int, int],
    cat_map_A: Dict[int, int],
    cat_map_B: Dict[int, int],
) -> List[Dict]:
    merged_annos: List[Dict] = []
    next_ann_id = 1

    for anno in cocoA_annos:
        new_anno = anno.copy()
        new_anno["id"] = next_ann_id
        new_anno["image_id"] = img_map_A[anno["image_id"]]
        new_anno["category_id"] = cat_map_A[anno["category_id"]]
        merged_annos.append(new_anno)
        next_ann_id += 1

    for anno in cocoB_annos:
        new_anno = anno.copy()
        new_anno["id"] = next_ann_id
        new_anno["image_id"] = img_map_B[anno["image_id"]]
        new_anno["category_id"] = cat_map_B[anno["category_id"]]
        merged_annos.append(new_anno)
        next_ann_id += 1

    return merged_annos


# ──────────────────────────────────────────────────────────────────────────────
# 8) Copy & rename images into the final “images/” folder.
#
#    - extractA_images, extractB_images: Paths to each dataset’s images/ dir
#    - merged_images: List[{"id": int, "file_name": str, …}] from step 6
#    - output_images: Path where we write the new “images/” folder
#
#    We’ll prefix “A_” or “B_” to avoid filename collisions.
# ──────────────────────────────────────────────────────────────────────────────
def copy_and_rename_images(
    extractA_images: Path,
    extractB_images: Path,
    merged_images: List[Dict],
    output_images: Path,
) -> None:
    output_images.mkdir(parents=True, exist_ok=True)

    # Helper: build a set of original A filenames for quick lookup
    filenames_A = {p.name for p in extractA_images.iterdir()} if extractA_images.exists() else set()

    for img_info in merged_images:
        # Determine whether this image came from A or B by filename
        fname = img_info["file_name"]
        if fname in filenames_A:
            src_folder = extractA_images
            prefix = "A_"
        else:
            src_folder = extractB_images
            prefix = "B_"

        src_path = src_folder / fname
        dst_fname = prefix + fname
        dst_path = output_images / dst_fname
        shutil.copyfile(src_path, dst_path)


# ──────────────────────────────────────────────────────────────────────────────
# 9) Save a COCO dict back to disk, creating parent folders as needed
# ──────────────────────────────────────────────────────────────────────────────
def save_coco_json(coco: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
