# -*- coding: utf-8 -*-
"""
Editor Spyder

Este é um arquivo de script temporário.
"""

from pathlib import Path

def delete_mp3s(data_dir="Data", dry_run=False):
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Folder not found: {data_path.resolve()}")

    mp3_files = list(data_path.rglob("*.mp3"))

    if not mp3_files:
        print(f"No .mp3 files found under: {data_path.resolve()}")
        return

    print(f"Found {len(mp3_files)} .mp3 file(s) under: {data_path.resolve()}\n")

    deleted = 0
    for f in mp3_files:
        try:
            if dry_run:
                print(f"[DRY RUN] Would delete: {f}")
            else:
                f.unlink()
                print(f"Deleted: {f}")
                deleted += 1
        except Exception as e:
            print(f"Failed to delete {f}: {e}")

    if dry_run:
        print("\nDry run complete. Set dry_run=False to actually delete files.")
    else:
        print(f"\nDone. Deleted {deleted} file(s).")

if __name__ == "__main__":
    # 1) Run once with dry_run=True to preview
    delete_mp3s("Data", dry_run=False)

    # 2) Then switch to dry_run=False to actually delete
    # delete_mp3s("Data", dry_run=False)
