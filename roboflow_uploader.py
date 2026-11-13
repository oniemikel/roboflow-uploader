import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow
from tqdm import tqdm


def upload_folders_as_batches(
    sample_dir: str,
    api_key: str,
    workspace_id: str,
    project_id: str,
    use_full_relpath_as_batch: bool = True,
):
    if not all([sample_dir, api_key, workspace_id, project_id]):
        raise ValueError("環境変数（SAMPLE_DIR, ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT）が不足しています。")

    root = Path(sample_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"フォルダが見つかりません: {root}")

    # Roboflow プロジェクトへ接続
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_id).project(project_id)

    target_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # 画像を含むサブフォルダをすべて収集
    folder_tasks = []
    for dirpath, _, filenames in sorted(os.walk(root)):
        current_dir = Path(dirpath)
        images = sorted(
            current_dir / f
            for f in filenames
            if Path(f).suffix.lower() in target_exts
        )
        if images:
            folder_tasks.append((current_dir, images))

    print(f"対象フォルダ（バッチ）数: {len(folder_tasks)}個 / 総画像数: {sum(len(imgs) for _, imgs in folder_tasks)}枚")

    for current_dir, images in tqdm(folder_tasks, desc="バッチアップロード進捗", unit="batch"):
        rel_path = current_dir.relative_to(root)

        # バッチ名の決定
        if rel_path == Path("."):
            batch_name = root.name
            tags = [root.name]
        else:
            # True: "親フォルダ_子フォルダ_動画名"（同名フォルダの衝突防止）
            # False: 末端のフォルダ名のみ（例: "video_01"）
            batch_name = "_".join(rel_path.parts) if use_full_relpath_as_batch else current_dir.name
            tags = list(rel_path.parts)  # 各階層のフォルダ名をタグとしても自動付与

        for img_path in images:
            project.upload(
                image_path=str(img_path),
                batch_name=batch_name,
                tag_names=tags,
                num_retry_uploads=3,
            )

    print("すべてのフォルダのアップロードが完了しました。")


if __name__ == "__main__":
    load_dotenv()

    upload_folders_as_batches(
        sample_dir=os.getenv("SAMPLE_DIR", "annotation_samples"),
        api_key=os.getenv("ROBOFLOW_API_KEY"),
        workspace_id=os.getenv("ROBOFLOW_WORKSPACE"),
        project_id=os.getenv("ROBOFLOW_PROJECT"),
        use_full_relpath_as_batch=True,
    )