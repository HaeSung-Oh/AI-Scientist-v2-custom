import os
import os.path as osp
import shutil
from dataclasses import dataclass


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_DIR_NAMES = {"image", "images", "img", "imgs", "original", "originals"}
MASK_DIR_NAMES = {
    "mask",
    "masks",
    "label",
    "labels",
    "annotation",
    "annotations",
    "ground truth",
    "ground_truth",
    "gt",
}

POLYP_DATASETS = {
    "Kvasir-SEG": {
        "aliases": ("kvasir", "kvasir-seg", "kvasir_seg"),
        "hf_repo": "Angelou0516/kvasir-seg",
        "min_pairs": 100,
    },
    "CVC-ClinicDB": {
        "aliases": ("clinic", "clinicdb", "cvc-clinicdb", "cvcclinicdb"),
        "hf_repo": "Angelou0516/CVC-ClinicDB",
        "min_pairs": 100,
    },
}

SYNTHETIC_MARKERS = (
    "class SyntheticDataset",
    "SyntheticDataset(",
    "torch.rand(",
    "np.random.rand(",
    "np.random.randn(",
    "\"synthetic\"",
    "'synthetic'",
)
REAL_DATA_MARKERS = (
    "input/Kvasir-SEG",
    "input/CVC-ClinicDB",
    "Kvasir-SEG",
    "CVC-ClinicDB",
)


@dataclass
class DatasetStatus:
    name: str
    path: str | None
    image_count: int = 0
    mask_count: int = 0

    @property
    def ready(self):
        spec = POLYP_DATASETS[self.name]
        return min(self.image_count, self.mask_count) >= spec["min_pairs"]


def count_images(path):
    if not path or not osp.isdir(path):
        return 0
    return sum(
        1
        for filename in os.listdir(path)
        if osp.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
    )


def walk_limited(search_root, max_depth=6):
    search_root = osp.abspath(search_root)
    base_depth = search_root.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(search_root):
        depth = root.rstrip(os.sep).count(os.sep) - base_depth
        if depth >= max_depth:
            dirs[:] = []
        yield root, dirs, files


def find_image_mask_pair(dataset_root):
    candidates = []
    for root, dirs, _ in walk_limited(dataset_root, max_depth=5):
        dir_map = {d.lower(): d for d in dirs}
        for image_name in IMAGE_DIR_NAMES:
            if image_name not in dir_map:
                continue
            for mask_name in MASK_DIR_NAMES:
                if mask_name not in dir_map:
                    continue
                image_dir = osp.join(root, dir_map[image_name])
                mask_dir = osp.join(root, dir_map[mask_name])
                image_count = count_images(image_dir)
                mask_count = count_images(mask_dir)
                if image_count and mask_count:
                    candidates.append((min(image_count, mask_count), image_dir, mask_dir))

    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1], candidates[0][2]


def dataset_status(name, dataset_root):
    pair = find_image_mask_pair(dataset_root)
    if pair is None:
        return DatasetStatus(name=name, path=None)
    image_dir, mask_dir = pair
    return DatasetStatus(
        name=name,
        path=dataset_root,
        image_count=count_images(image_dir),
        mask_count=count_images(mask_dir),
    )


def find_dataset_root(search_root, dataset_name, aliases, max_depth=6):
    if not search_root or not osp.isdir(search_root):
        return None

    matches = []
    for root, dirs, _ in walk_limited(search_root, max_depth=max_depth):
        base = osp.basename(root).lower()
        if any(alias in base for alias in aliases):
            status = dataset_status(dataset_name, root)
            if status.ready:
                matches.append((min(status.image_count, status.mask_count), root))
        for dirname in dirs:
            dirname_lower = dirname.lower()
            if any(alias in dirname_lower for alias in aliases):
                candidate = osp.join(root, dirname)
                status = dataset_status(dataset_name, candidate)
                if status.ready:
                    matches.append((min(status.image_count, status.mask_count), candidate))

    if not matches:
        return None
    matches.sort(reverse=True, key=lambda item: item[0])
    print(f"Found {dataset_name} candidate at {matches[0][1]}")
    return matches[0][1]


def link_or_copytree(src, dst, use_symlinks=True):
    if osp.lexists(dst):
        if osp.islink(dst) or osp.isfile(dst):
            os.unlink(dst)
        else:
            shutil.rmtree(dst)
    if use_symlinks:
        os.symlink(osp.abspath(src), dst)
    else:
        shutil.copytree(src, dst)


def prepare_dataset_pair(src_root, dst_root, dataset_name, use_symlinks=True):
    pair = find_image_mask_pair(src_root)
    if pair is None:
        raise RuntimeError(
            f"Could not find image/mask folders for {dataset_name} under {src_root}."
        )
    image_dir, mask_dir = pair
    dataset_dst = osp.join(dst_root, dataset_name)
    if osp.lexists(dataset_dst):
        if osp.islink(dataset_dst) or osp.isfile(dataset_dst):
            os.unlink(dataset_dst)
        else:
            shutil.rmtree(dataset_dst)
    os.makedirs(dataset_dst, exist_ok=True)
    link_or_copytree(image_dir, osp.join(dataset_dst, "images"), use_symlinks)
    link_or_copytree(mask_dir, osp.join(dataset_dst, "masks"), use_symlinks)
    status = dataset_status(dataset_name, dataset_dst)
    print(
        f"Prepared {dataset_name}: "
        f"{status.image_count} images, {status.mask_count} masks "
        f"({'symlinked' if use_symlinks else 'copied'})"
    )
    return status


def download_hf_dataset(repo_id, download_root):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "Downloading datasets requires huggingface_hub. Install it with "
            "`conda run -n ai_scientist python -m pip install huggingface_hub`."
        ) from e

    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=osp.join(download_root, repo_id.replace("/", "__")),
        local_dir_use_symlinks=False,
    )


def default_dataset_search_roots():
    roots = []
    candidates = [
        os.environ.get("POLYP_DATASET_ROOT"),
        os.environ.get("AI_SCIENTIST_DATA_ROOT"),
        osp.expanduser("~/Polyp"),
        osp.expanduser("~/datasets"),
        osp.expanduser("~/data"),
        osp.expanduser("~/BP"),
        osp.abspath(osp.join(os.getcwd(), "..")),
        osp.abspath(os.getcwd()),
    ]

    for candidate in candidates:
        if candidate and osp.isdir(candidate) and candidate not in roots:
            roots.append(candidate)
    return roots


def prepare_real_polyp_data(
    idea_dir,
    dataset_root=None,
    auto_discover=True,
    auto_download=True,
    force_download=False,
    allow_missing=False,
    use_symlinks=True,
):
    data_dir = osp.join(idea_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    source_roots = []
    if dataset_root:
        source_roots.append(osp.abspath(dataset_root))

    if auto_discover:
        discovered_roots = default_dataset_search_roots()
        print("Searching for local polyp datasets in:")
        for root in discovered_roots:
            print(f"  - {root}")
        source_roots.extend(root for root in discovered_roots if root not in source_roots)

    missing = []
    statuses = {}
    for dataset_name, spec in POLYP_DATASETS.items():
        prepared_path = osp.join(data_dir, dataset_name)
        status = dataset_status(dataset_name, prepared_path)
        if status.ready:
            statuses[dataset_name] = status
            continue

        source_root = None
        for root in source_roots:
            source_root = find_dataset_root(root, dataset_name, spec["aliases"])
            if source_root:
                break
        if source_root:
            statuses[dataset_name] = prepare_dataset_pair(
                source_root, data_dir, dataset_name, use_symlinks=use_symlinks
            )
        else:
            missing.append(dataset_name)

    should_download = force_download or auto_download
    if missing and should_download:
        download_root = osp.join(idea_dir, "_downloaded_datasets")
        os.makedirs(download_root, exist_ok=True)
        still_missing = []
        for dataset_name in missing:
            spec = POLYP_DATASETS[dataset_name]
            print(f"Downloading {dataset_name} from Hugging Face: {spec['hf_repo']}")
            downloaded_root = download_hf_dataset(spec["hf_repo"], download_root)
            source_root = find_dataset_root(
                downloaded_root, dataset_name, spec["aliases"], max_depth=8
            )
            if source_root:
                statuses[dataset_name] = prepare_dataset_pair(
                    source_root, data_dir, dataset_name, use_symlinks=use_symlinks
                )
            else:
                still_missing.append(dataset_name)
        missing = still_missing

    if missing and not allow_missing:
        raise RuntimeError(
            "Missing required real polyp datasets: "
            + ", ".join(missing)
            + ". Provide them with `--dataset-root /path/to/data` or use "
            "`--download-polyp-data`. By default the launcher searches local "
            "dataset folders and attempts public downloads; use "
            "`--no-auto-download-polyp-data` to disable downloads. Synthetic "
            "data is not accepted for final validation."
        )

    return data_dir, statuses


def code_uses_synthetic_data(code):
    code_lower = code.lower()
    has_synthetic_marker = any(marker.lower() in code_lower for marker in SYNTHETIC_MARKERS)
    has_real_marker = any(marker.lower() in code_lower for marker in REAL_DATA_MARKERS)
    return has_synthetic_marker and not has_real_marker
