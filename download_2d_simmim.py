import argparse
import os
import urllib.request

from project_config import SAVE_DIR


PRETRAINED_2D_DIR = os.path.join(SAVE_DIR, "pretrained_2d")

HF_SIMMIM_MODELS = {
    "swin-base-simmim-window6-192": {
        "repo_id": "microsoft/swin-base-simmim-window6-192",
        "filename": "pytorch_model.bin",
        "out_name": "swin_base_simmim_window6_192.pth",
    },
    "swin-large-simmim-window12-192": {
        "repo_id": "microsoft/swin-large-simmim-window12-192",
        "filename": "pytorch_model.bin",
        "out_name": "swin_large_simmim_window12_192.pth",
    },
}


def hf_resolve_url(repo_id, filename):
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def download_with_resume(url, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    temp_path = out_path + ".part"
    resume_at = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

    request = urllib.request.Request(url)
    if resume_at > 0:
        request.add_header("Range", f"bytes={resume_at}-")

    mode = "ab" if resume_at > 0 else "wb"
    print(f"[INFO] downloading: {url}")
    if resume_at > 0:
        print(f"[INFO] resuming from byte offset: {resume_at}")

    with urllib.request.urlopen(request) as response, open(temp_path, mode) as f:
        total = response.headers.get("Content-Length")
        total = int(total) + resume_at if total is not None else None
        downloaded = resume_at
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100.0 * downloaded / max(total, 1)
                print(f"\r[INFO] {downloaded / 1024 ** 2:.1f} / {total / 1024 ** 2:.1f} MB ({pct:.1f}%)", end="")
            else:
                print(f"\r[INFO] {downloaded / 1024 ** 2:.1f} MB", end="")
    print()
    os.replace(temp_path, out_path)
    return out_path


def download_with_hf_hub(repo_id, filename, out_path):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cached = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=os.path.dirname(out_path))
    if os.path.abspath(cached) != os.path.abspath(out_path):
        with open(cached, "rb") as src, open(out_path, "wb") as dst:
            dst.write(src.read())
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download 2D SimMIM/Swin pretrained weights.")
    parser.add_argument(
        "--model",
        default="swin-base-simmim-window6-192",
        choices=sorted(HF_SIMMIM_MODELS.keys()),
        help="Known Hugging Face SimMIM checkpoint to download.",
    )
    parser.add_argument("--url", default=None, help="Custom checkpoint URL. Overrides --model.")
    parser.add_argument("--out", default=None, help="Output .pth/.bin path.")
    parser.add_argument("--no-hf-hub", action="store_true", help="Use urllib even if huggingface_hub is installed.")
    args = parser.parse_args()

    if args.url:
        url = args.url
        out_name = os.path.basename(url.split("?", 1)[0]) or "simmim_2d_pretrained.pth"
        out_path = args.out or os.path.join(PRETRAINED_2D_DIR, out_name)
        download_with_resume(url, out_path)
    else:
        spec = HF_SIMMIM_MODELS[args.model]
        out_path = args.out or os.path.join(PRETRAINED_2D_DIR, spec["out_name"])
        if not args.no_hf_hub:
            downloaded = download_with_hf_hub(spec["repo_id"], spec["filename"], out_path)
            if downloaded is not None:
                print(f"[INFO] downloaded with huggingface_hub: {downloaded}")
                return
        url = hf_resolve_url(spec["repo_id"], spec["filename"])
        download_with_resume(url, out_path)

    size_mb = os.path.getsize(out_path) / 1024 ** 2
    print(f"[INFO] saved: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
