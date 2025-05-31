# utils.py

import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ──────────────────────────────────────────────────────────────────────────────
# 1) Load the LLaMA 2 chat model and tokenizer once at module import time.
#    We assume you have already downloaded "meta-llama/Llama-2-7b-chat" locally
#    or are able to pull it from Hugging Face. If you want to point to a
#    local folder, replace the string below with that path.
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "meta-llama/Llama-2-7b-chat"

# Device selection: use GPU if available, else CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
)
model.to(DEVICE)
model.eval()

# ──────────────────────────────────────────────────────────────────────────────
# 2) Replace OpenAI‐based get_label_mapping(...) with LLaMA 2 inference.
#
#    We structure the prompt in the “LLaMA 2 chat” format:
#      <s>[INST] <<SYS>> ...system prompt... <</SYS>>
#      user prompt [/INST]
#    Then generate and parse the JSON output.
# ──────────────────────────────────────────────────────────────────────────────

def get_label_mapping(names_A: list[str], names_B: list[str]) -> dict[str, str | None]:
    """
    Ask LLaMA 2‐chat to return a JSON mapping from each name in names_A to its best match in names_B.
    If no match is found, the value should be null.
    """
    system_prompt = "You are a helpful assistant that maps category names from List A to best matches in List B."
    user_prompt = (
        f"List A: {names_A}\n"
        f"List B: {names_B}\n\n"
        "Please output a JSON object where each key is a name from List A and its value is the best match from List B.\n"
        "If no good match exists, set the value to null.\n"
        "Be case‐insensitive. For example:\n"
        '{ "red": "r", "paper": "Paper", "foo": null }\n'
        "Provide only the JSON object in your response."
    )

    # Build the LLaMA 2 “chat” style prompt
    full_prompt = (
        "<s>[INST] <<SYS>>\n"
        + system_prompt
        + "\n<</SYS>>\n"
        + user_prompt
        + " [/INST]"
    )

    # Tokenize & run generation
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length - 1,
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    # Generate up to 512 new tokens (adjust if your JSON is very large)
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        temperature=0.0,      # deterministic
        do_sample=False,      # greedy
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode output (skip special tokens)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # The model’s response will include everything after the prompt.
    # We need to extract the JSON substring.
    # Often, the JSON begins with '{' and ends with '}', so find the first '{' and last '}'.
    try:
        first_brace = decoded.index("{")
        last_brace = decoded.rindex("}")
        json_str = decoded[first_brace : last_brace + 1]
    except ValueError:
        # Fallback: assume entire decoded is JSON
        json_str = decoded.strip()

    # Parse JSON
    mapping = json.loads(json_str)
    # Convert any explicit "null" → None
    for k, v in list(mapping.items()):
        if v is None:
            mapping[k] = None
        elif isinstance(v, str) and v.lower() == "null":
            mapping[k] = None

    return mapping


# ──────────────────────────────────────────────────────────────────────────────
# 3) The remainder of utils.py is unchanged from before:
#    – load_coco_json
#    – extract_categories
#    – build_unified_categories
#    – remap_images
#    – remap_annotations
#    – copy_and_rename_images
#    – save_coco_json
# ──────────────────────────────────────────────────────────────────────────────

def load_coco_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_categories(coco: dict) -> dict[int, tuple[str, str]]:
    return {
        cat["id"]: (cat["name"], cat.get("supercategory", ""))
        for cat in coco.get("categories", [])
    }


def build_unified_categories(
    cats_A: dict[int, tuple[str, str]],
    cats_B: dict[int, tuple[str, str]],
    name_map_A_to_B: dict[str, str | None],
) -> tuple[list[dict], dict[int, int], dict[int, int]]:
    unified = []
    remap_A = {}
    remap_B = {}
    next_id = 1

    for old_id_A, (nameA, superA) in cats_A.items():
        matchedB = name_map_A_to_B.get(nameA)
        if matchedB:
            candidates = [
                bid
                for bid, (nB, _) in cats_B.items()
                if nB.lower() == matchedB.lower()
            ]
            if candidates:
                old_id_B = candidates[0]
                if old_id_B in remap_B:
                    new_id = remap_B[old_id_B]
                    remap_A[old_id_A] = new_id
                    continue
                unified.append({"id": next_id, "name": nameA, "supercategory": superA})
                remap_A[old_id_A] = next_id
                remap_B[old_id_B] = next_id
                next_id += 1
            else:
                unified.append({"id": next_id, "name": nameA, "supercategory": superA})
                remap_A[old_id_A] = next_id
                next_id += 1
        else:
            unified.append({"id": next_id, "name": nameA, "supercategory": superA})
            remap_A[old_id_A] = next_id
            next_id += 1

    for old_id_B, (nameB, superB) in cats_B.items():
        if old_id_B in remap_B:
            continue
        unified.append({"id": next_id, "name": nameB, "supercategory": superB})
        remap_B[old_id_B] = next_id
        next_id += 1

    return unified, remap_A, remap_B


def remap_images(
    images_A: list[dict],
    images_B: list[dict],
) -> tuple[list[dict], dict[int, int], dict[int, int]]:
    merged_images = []
    image_map_A = {}
    image_map_B = {}
    next_img_id = 1

    for img in images_A:
        old_id = img["id"]
        new_id = next_img_id
        image_map_A[old_id] = new_id
        new_entry = img.copy()
        new_entry["id"] = new_id
        orig_name = img["file_name"]
        new_fname = f"A_{orig_name}"
        new_entry["file_name"] = new_fname
        new_entry["source"] = "A"
        new_entry["orig_file_name"] = orig_name
        merged_images.append(new_entry)
        next_img_id += 1

    for img in images_B:
        old_id = img["id"]
        new_id = next_img_id
        image_map_B[old_id] = new_id
        new_entry = img.copy()
        new_entry["id"] = new_id
        orig_name = img["file_name"]
        new_fname = f"B_{orig_name}"
        new_entry["file_name"] = new_fname
        new_entry["source"] = "B"
        new_entry["orig_file_name"] = orig_name
        merged_images.append(new_entry)
        next_img_id += 1

    return merged_images, image_map_A, image_map_B


def remap_annotations(
    annos_A: list[dict],
    annos_B: list[dict],
    image_map_A: dict[int, int],
    image_map_B: dict[int, int],
    cat_map_A: dict[int, int],
    cat_map_B: dict[int, int],
) -> list[dict]:
    merged_annos = []
    next_anno_id = 1

    def rewrite(anno: dict, new_img: int, new_cat: int, new_id: int) -> dict:
        new_anno = anno.copy()
        new_anno["id"] = new_id
        new_anno["image_id"] = new_img
        new_anno["category_id"] = new_cat
        return new_anno

    for anno in annos_A:
        old_img = anno["image_id"]
        old_cat = anno["category_id"]
        new_img = image_map_A[old_img]
        new_cat = cat_map_A[old_cat]
        merged_annos.append(rewrite(anno, new_img, new_cat, next_anno_id))
        next_anno_id += 1

    for anno in annos_B:
        old_img = anno["image_id"]
        old_cat = anno["category_id"]
        new_img = image_map_B[old_img]
        new_cat = cat_map_B[old_cat]
        merged_annos.append(rewrite(anno, new_img, new_cat, next_anno_id))
        next_anno_id += 1

    return merged_annos


def copy_and_rename_images(
    extractA_images: Path,
    extractB_images: Path,
    merged_images: list[dict],
    dest_images_dir: Path,
) -> None:
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    for img in merged_images:
        source = img["source"]
        orig_name = img["orig_file_name"]
        new_name = img["file_name"]
        if source == "A":
            src_path = extractA_images / orig_name
        else:
            src_path = extractB_images / orig_name
        shutil.copy(src_path, dest_images_dir / new_name)


def save_coco_json(merged: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)