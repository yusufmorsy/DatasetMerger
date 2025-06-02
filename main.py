import logging
import shutil
import zipfile
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv
load_dotenv() 

from utils import (
    load_coco_json,
    extract_categories,
    get_label_mapping,
    build_unified_categories,
    remap_images,
    remap_annotations,
    copy_and_rename_images,
    save_coco_json,
)

# ──────────────────────────────
# Logging setup
# ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,                      # change to DEBUG for super-verbose output
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("dataset-merger")

# ──────────────────────────────
# FastAPI app
# ──────────────────────────────
app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
def index(request: Request):
    log.info("Index page served")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/merge")
async def merge_datasets(
    fileA: UploadFile = File(...),
    fileB: UploadFile = File(...),
):
    log.info("⇨ /merge called – starting merge")
    tmpdir = Path(tempfile.mkdtemp())
    zipA_path = tmpdir / "A.zip"
    zipB_path = tmpdir / "B.zip"

    # 1 - Save uploaded ZIPs
    log.info("Step 1 – saving uploads")
    zipA_path.write_bytes(await fileA.read())
    zipB_path.write_bytes(await fileB.read())
    log.debug("  ↳ %s (%d B) | %s (%d B)",
              zipA_path.name, zipA_path.stat().st_size,
              zipB_path.name, zipB_path.stat().st_size)

    # 2 - Extract each ZIP
    log.info("Step 2 – extracting ZIPs")
    extractA, extractB = tmpdir / "A", tmpdir / "B"
    extractA.mkdir(); extractB.mkdir()
    with zipfile.ZipFile(zipA_path) as zfA:
        zfA.extractall(extractA)
    with zipfile.ZipFile(zipB_path) as zfB:
        zfB.extractall(extractB)

    # 3 - Locate *_annotations.coco.json
    log.info("Step 3 – locating COCO annotation files")
    cocoA_files = list(extractA.rglob("*_annotations.coco.json"))
    cocoB_files = list(extractB.rglob("*_annotations.coco.json"))
    if not cocoA_files or not cocoB_files:
        log.error("No annotation JSON found in one or both uploads")
        raise HTTPException(
            status_code=400,
            detail="Could not find any '*_annotations.coco.json' in one or both uploads.",
        )
    cocoA_path, cocoB_path = cocoA_files[0], cocoB_files[0]
    log.debug("  ↳ %s | %s", cocoA_path.relative_to(extractA), cocoB_path.relative_to(extractB))

    # 4 - Load JSON
    log.info("Step 4 – loading COCO JSON files")
    cocoA, cocoB = load_coco_json(cocoA_path), load_coco_json(cocoB_path)

    # 5 - Extract categories
    log.info("Step 5 – extracting category dictionaries")
    catsA, catsB = extract_categories(cocoA), extract_categories(cocoB)

    # 6 - Call LLaMA for label mapping
    log.info("Step 6 – calling GPT 3.5 Turbo for label mapping")
    names_A = [n for (_, (n, _)) in catsA.items()]
    names_B = [n for (_, (n, _)) in catsB.items()]
    name_map_A_to_B = get_label_mapping(names_A, names_B)
    log.debug("  ↳ label mapping: %s", name_map_A_to_B)

    # 7 - Build unified categories
    log.info("Step 7 – building unified category list")
    unified_cats, remap_A_cats, remap_B_cats = build_unified_categories(
        catsA, catsB, name_map_A_to_B
    )

    # 8 - Remap image IDs
    log.info("Step 8 – remapping image IDs")
    merged_images, img_id_map_A, img_id_map_B = remap_images(
        cocoA["images"], cocoB["images"]
    )

    # 9 - Remap & merge annotations
    log.info("Step 9 – remapping annotations")
    merged_annotations = remap_annotations(
        cocoA["annotations"], cocoB["annotations"],
        img_id_map_A, img_id_map_B,
        remap_A_cats, remap_B_cats,
    )

    # 10 - Copy & rename images
    log.info("Step 10 – copying image files into merged_dataset/images")
    output_folder = tmpdir / "merged_dataset"
    output_images = output_folder / "images"
    annotations_folder = output_folder / "annotations"
    output_folder.mkdir(parents=True, exist_ok=True)
    annotations_folder.mkdir(parents=True, exist_ok=True)
    # 10 - Copy & rename images (new):
    copy_and_rename_images(
        extractA,       # search entire extraction folder A
        extractB,       # search entire extraction folder B
        merged_images,
        output_images,
    )

    # 11 - Write merged COCO JSON
    log.info("Step 11 – writing merged_annotations.coco.json")
    merged_coco = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": unified_cats,
    }
    merged_json_path = annotations_folder / "merged_annotations.coco.json"
    save_coco_json(merged_coco, merged_json_path)

    # 12 - Zip the final dataset
    log.info("Step 12 – zipping final dataset")
    zip_path = tmpdir / "merged_dataset.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", output_folder)

    log.info("✓ Merge complete – sending merged_dataset.zip")
    return FileResponse(
        path=str(zip_path),
        filename="merged_dataset.zip",
        media_type="application/zip",
    )
