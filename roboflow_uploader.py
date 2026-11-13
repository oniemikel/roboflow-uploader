import os
from pathlib import Path
import requests
from dotenv import load_dotenv
from roboflow import Roboflow
from tqdm import tqdm


def fetch_existing_batch_names(
    project, api_key: str, workspace_id: str | None, project_id: str
) -> set[str]:
    """Roboflowプロジェクトにすでに存在するバッチ名の一覧を取得する"""
    try:
        # Roboflow Python SDK の公式メソッドで取得
        if hasattr(project, "get_batches"):
            res = project.get_batches()
        else:
            # SDKバージョンが古い場合のREST APIフォールバック
            ws = workspace_id or getattr(project, "workspace", None)
            url = f"https://api.roboflow.com/{ws}/{project_id}/batches"
            res = requests.get(url, params={"api_key": api_key}, timeout=15).json()

        batches = res.get("batches", []) if isinstance(res, dict) else []
        return {b["name"] for b in batches if b.get("name")}
    except Exception as e:
        print(
            f"[警告] 既存バッチ一覧の取得に失敗したため、重複チェックなしで続行します: {e}"
        )
        return set()


def upload_folders_as_batches(
    sample_dir: str,
    api_key: str,
    project_id: str,
    workspace_id: str | None = None,
    use_full_relpath_as_batch: bool = True,
):
    if not all([sample_dir, api_key, project_id]):
        raise ValueError(
            "環境変数（SAMPLE_DIR, ROBOFLOW_API_KEY, ROBOFLOW_PROJECT）が不足しています。"
        )

    root = Path(sample_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"フォルダが見つかりません: {root}")

    # Roboflow プロジェクトへ接続（workspace_id が空ならAPIキーから自動取得）
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace_id) if workspace_id else rf.workspace()
    project = ws.project(project_id)

    # 既存バッチ名を取得
    existing_batches = fetch_existing_batch_names(
        project, api_key, workspace_id, project_id
    )
    print(f"Roboflow上の既存バッチ数: {len(existing_batches)}個")

    target_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # 画像を含むサブフォルダを収集し、新規と既存に振り分け
    upload_tasks = []
    skipped_batches = []

    for dirpath, _, filenames in sorted(os.walk(root)):
        current_dir = Path(dirpath)
        images = sorted(
            current_dir / f for f in filenames if Path(f).suffix.lower() in target_exts
        )
        if not images:
            continue

        rel_path = current_dir.relative_to(root)
        if rel_path == Path("."):
            batch_name = root.name
            tags = [root.name]
        else:
            batch_name = (
                "_".join(rel_path.parts)
                if use_full_relpath_as_batch
                else current_dir.name
            )
            tags = list(rel_path.parts)

        # すでに同じバッチ名が存在する場合は除外
        if batch_name in existing_batches:
            skipped_batches.append(batch_name)
            continue

        upload_tasks.append((batch_name, tags, images))

    print(
        f"スキップ（既存）: {len(skipped_batches)}バッチ / "
        f"新規アップロード対象: {len(upload_tasks)}バッチ（計 {sum(len(imgs) for _, _, imgs in upload_tasks)}枚）"
    )

    if not upload_tasks:
        print("新規にアップロードするバッチはありません。")
        return

    for batch_name, tags, images in tqdm(
        upload_tasks, desc="新規バッチアップロード", unit="batch"
    ):
        for img_path in images:
            project.upload(
                image_path=str(img_path),
                batch_name=batch_name,
                tag_names=tags,
                num_retry_uploads=3,
            )

    print("すべての新規バッチのアップロードが完了しました。")


if __name__ == "__main__":
    load_dotenv()

    upload_folders_as_batches(
        sample_dir=os.getenv("SAMPLE_DIR", "annotation_samples"),
        api_key=os.getenv("ROBOFLOW_API_KEY"),
        project_id=os.getenv("ROBOFLOW_PROJECT"),
        workspace_id=os.getenv("ROBOFLOW_WORKSPACE"),  # 未設定でも自動取得されます
        use_full_relpath_as_batch=True,
    )
