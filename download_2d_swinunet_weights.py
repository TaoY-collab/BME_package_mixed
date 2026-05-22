import argparse
import os
import urllib.request

from project_config import SAVE_DIR


PRETRAINED_2D_DIR = os.path.join(SAVE_DIR, "pretrained_2d")
DEFAULT_OUT = os.path.join(PRETRAINED_2D_DIR, "swin_tiny_patch4_window7_224.pth")

SOURCES = {
    "official": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
    "hf_mirror": "https://huggingface.co/ranksu/deit_tiny_model_ckpt/resolve/main/swin_tiny_patch4_window7_224.pth",
}


def download_with_resume(url, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    part_path = out_path + ".part"
    resume_at = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    request = urllib.request.Request(url)
    if resume_at > 0:
        request.add_header("Range", f"bytes={resume_at}-")
        print(f"[INFO] resume from byte offset: {resume_at}")

    mode = "ab" if resume_at > 0 else "wb"
    print(f"[INFO] downloading Swin-Unet 2D pretrained backbone: {url}")
    with urllib.request.urlopen(request) as response, open(part_path, mode) as f:
        content_length = response.headers.get("Content-Length")
        total = int(content_length) + resume_at if content_length is not None else None
        downloaded = resume_at
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(
                    f"\r[INFO] {downloaded / 1024 ** 2:.1f}/{total / 1024 ** 2:.1f} MB "
                    f"({100.0 * downloaded / max(total, 1):.1f}%)",
                    end="",
                )
            else:
                print(f"\r[INFO] {downloaded / 1024 ** 2:.1f} MB", end="")
    print()
    os.replace(part_path, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download the 2D Swin-Unet pretrained Swin-T backbone checkpoint "
            "swin_tiny_patch4_window7_224.pth."
        )
    )
    parser.add_argument("--source", choices=sorted(SOURCES.keys()), default="official")
    parser.add_argument("--url", default=None, help="Custom checkpoint URL. Overrides --source.")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output path. Default: {DEFAULT_OUT}")
    parser.add_argument("--force", action="store_true", help="Redownload even if output already exists.")
    args = parser.parse_args()

    url = args.url or SOURCES[args.source]
    if os.path.exists(args.out) and not args.force:
        size_mb = os.path.getsize(args.out) / 1024 ** 2
        print(f"[INFO] checkpoint already exists: {args.out} ({size_mb:.1f} MB)")
        return

    out_path = download_with_resume(url, args.out)
    size_mb = os.path.getsize(out_path) / 1024 ** 2
    print(f"[INFO] saved: {out_path} ({size_mb:.1f} MB)")
    print("[INFO] expected Swin-Unet config path: pretrained_ckpt/swin_tiny_patch4_window7_224.pth")


if __name__ == "__main__":
    main()
