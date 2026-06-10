import base64
import tempfile
import runpod

from app.workflow import graph
from app.services.fraud_detection import detect_fraud
from app.services.vqa import get_vqa_model
from app.services.stt import transcribe_pipeline


# -----------------------------
# Helpers
# -----------------------------
def _strip_base64_header(data: str) -> str:
    """Removes data:*base64, prefix if present."""
    if isinstance(data, str):
        if "base64," in data:
            return data.split("base64,", 1)[1]
        if data.startswith("data:") and "," in data:
            return data.split(",", 1)[1]
    return data


def _decode_to_temp_file(b64_data: str, suffix: str, temp_files: list):
    """Decode base64 and store in temp file."""
    raw = _strip_base64_header(b64_data)
    file_bytes = base64.b64decode(raw)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.close()

    temp_files.append(tmp.name)
    return tmp.name


# -----------------------------
# Handler
# -----------------------------
def handler(job):
    job_input = job.get("input", {})
    action = job_input.get("action", "full_pipeline")

    temp_files = []

    # =========================
    # 🔧 BASE64 NORMALIZATION
    # =========================

    voice_path = None
    if "voice_base64" in job_input:
        try:
            voice_path = _decode_to_temp_file(
                job_input["voice_base64"],
                ".mp3",
                temp_files
            )
        except Exception as e:
            return {"error": f"Failed to decode voice audio: {str(e)}"}

    images = []
    if "images_base64" in job_input:
        imgs = job_input["images_base64"]
        if not isinstance(imgs, list):
            imgs = [imgs]

        for idx, img_b64 in enumerate(imgs):
            try:
                img_path = _decode_to_temp_file(
                    img_b64,
                    ".jpg",
                    temp_files
                )

                images.append({
                    "image_id": f"IMG-{idx+1:03d}",
                    "image_path": img_path,
                    "ocr_extracted_text": ""
                })

            except Exception as e:
                # don't fail entire pipeline for one bad image
                print(f"[WARN] Image {idx} failed: {e}")

    # =========================
    # ROUTING
    # =========================

    if action == "full_pipeline":
        initial_state = {
            "text": job_input.get("text", ""),
            "images": images,
            "voice_path": voice_path,
            "evidence": {},
            "inquiry_history": [],
            "loop_count": 0
        }
        return graph.invoke(initial_state)

    elif action == "fraud_check":
        image_path = images[0]["image_path"] if images else None
        return detect_fraud(image_path)

    elif action == "vqa":
        vqa_model, vqa_processor = get_vqa_model()
        image_path = images[0]["image_path"] if images else None
        return {"status": "vqa_done", "image_used": image_path}

    elif action == "stt":
        return {
            "text": transcribe_pipeline(voice_path)
        }

    else:
        return {"error": f"Action '{action}' not supported."}


runpod.serverless.start({"handler": handler})