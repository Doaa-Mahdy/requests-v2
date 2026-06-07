import runpod
from app.workflow import graph
from app.services.fraud_detection import detect_fraud
from app.services.vqa import get_vqa_model
from app.services.stt import transcribe_pipeline

def handler(job):
    job_input = job.get("input", {})
    action = job_input.get("action", "full_pipeline") # Default to full
    
    # Route to specific logic based on 'action'
    if action == "full_pipeline":
        initial_state = {
            "text": job_input.get("text", ""),
            "images": job_input.get("images", []),
            "voice_path": job_input.get("audio_path"),
            "evidence": {},
            "inquiry_history": [],
            "loop_count": 0
        }
        return graph.invoke(initial_state)

    elif action == "fraud_check":
        return detect_fraud(job_input.get("image_path"))

    elif action == "vqa":
        # Simplified VQA route
        vqa_model, vqa_processor = get_vqa_model()
        # logic to run vqa on single image...
        return {"status": "vqa_done"}

    elif action == "stt":
        return {"text": transcribe_pipeline(job_input.get("audio_path"))}

    else:
        return {"error": f"Action '{action}' not supported."}

runpod.serverless.start({"handler": handler})