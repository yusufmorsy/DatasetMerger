import shutil
import zipfile
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

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

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
def index(request: Request):
    """
    Render the index.html template with a file‐upload form.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/merge")
async def merge_datasets(
    fileA: UploadFile = File(...),
    fileB: UploadFile = File(...),
):
    """
    Receive two uploaded ZIPs (each containing a COCO JSON + images/ folder),
    merge them into one COCO dataset, then return a merged ZIP.
    """
    tmpdir = Path(tempfile.mkdtemp())
    zipA_path = tmpdir / "A.zip"
    zipB_path = tmpdir / "B.zip"

    # 1) Save uploaded ZIPs to disk
    with open(zipA_path, "wb") as f:
        f.write(await fileA.read())
    with open(zipB_path, "wb") as f:
        f.write(await fileB.read())

    # 2) Extract each ZIP into its own folder
    extractA = tmpdir / "A"
    extractB = tmpdir / "B"
    extractA.mkdir()
    extractB.mkdir()
    with zipfile.ZipFile(zipA_path, "r") as zfA:
        zfA.extractall(extractA)
    with zipfile.ZipFile(zipB_path, "r") as zfB:
        zfB.extractall(extractB)

    # 3) Find the first COCO JSON in each extraction
    cocoA_files = list(extractA.rglob("*_annotations.coco.json"))
    cocoB_files = list(extractB.rglob("*_annotations.coco.json"))
    if not cocoA_files or not cocoB_files:
        raise HTTPException(
            status_code=400,
            detail="Could not find any '*_annotations.coco.json' in one or both uploads.",
        )
    cocoA_path = cocoA_files[0]
    cocoB_path = cocoB_files[0]

    # 4) Load the COCO JSONs
    cocoA = load_coco_json(cocoA_path)
    cocoB = load_coco_json(cocoB_path)

    # 5) Extract categories as {id: (name, supercategory)}
    catsA = extract_categories(cocoA)
    catsB = extract_categories(cocoB)

    # 6) Build name lists and ask LLaMA 2 to match them
    names_A = [name for (_, (name, _)) in catsA.items()]
    names_B = [name for (_, (name, _)) in catsB.items()]
    name_map_A_to_B = get_label_mapping(names_A, names_B)

    # 7) Build unified categories and get remapping dicts
    unified_cats, remap_A_cats, remap_B_cats = build_unified_categories(
        catsA, catsB, name_map_A_to_B
    )

    # 8) Remap image IDs (create merged_images list and maps)
    merged_images, img_id_map_A, img_id_map_B = remap_images(
        cocoA["images"], cocoB["images"]
    )

    # 9) Remap & merge annotations using the ID maps
    merged_annotations = remap_annotations(
        cocoA["annotations"],
        cocoB["annotations"],
        img_id_map_A,
        img_id_map_B,
        remap_A_cats,
        remap_B_cats,
    )

    # 10) Copy & rename actual image files into a new output folder
    output_folder = tmpdir / "merged_dataset"
    output_images = output_folder / "images"
    annotations_folder = output_folder / "annotations"

    # Create necessary folders
    output_folder.mkdir(parents=True, exist_ok=True)
    annotations_folder.mkdir(parents=True, exist_ok=True)
    copy_and_rename_images(
        extractA / "images", extractB / "images", merged_images, output_images
    )

    # 11) Write the merged COCO JSON
    merged_coco = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": unified_cats,
    }
    merged_json_path = annotations_folder / "merged_annotations.coco.json"
    save_coco_json(merged_coco, merged_json_path)

    # 12) ZIP the final merged_dataset folder
    zip_path = tmpdir / "merged_dataset.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", output_folder)

    # 13) Return the ZIP as a downloadable file
    return FileResponse(
        path=str(zip_path),
        filename="merged_dataset.zip",
        media_type="application/zip",
    )
