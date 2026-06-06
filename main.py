from app.workflow import graph

if __name__ == "__main__":

    sample = {
        "request_id": "REQ-001",
        "text": "محتاج علاج فيروس سي ومش قادر أتحمل التكلفة",
        "voice_path": None,
        "images": ["data/img15.jpg"]
    }

    result = graph.invoke(sample)

    print(result["final_output"])