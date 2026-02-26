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
        "reasoning": f"Analyzed hazard with context: {request.context}. The primary path is severely blocked. Re-routing to the safest adjacent vector.",
        "action": "ROUTE_OVERRIDE",
        "vector_x": 15.0,
        "vector_y": 5.0,
        "altitude_adjustment": 2.5
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
