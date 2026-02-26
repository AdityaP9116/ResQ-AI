from fastapi import FastAPI
from pydantic import BaseModel
import time

app = FastAPI()

class AnalyzeRequest(BaseModel):
    image_base64: str
    context: str

@app.post("/analyze")
async def analyze_image(request: AnalyzeRequest):
    print("Received analyze request.")
    # Simulate VLM processing time
    time.sleep(2)
    
    # In the future, this will run Cosmos-Reason2-2B
    return {
        "status": "critical",
        "advice": f"Analyzed hazard with context: {request.context}. Proceed with caution.",
        "hazard_details": "Debris blocking the main pathway."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
