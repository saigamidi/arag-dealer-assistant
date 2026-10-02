from fastapi import FastAPI

app = FastAPI(title="ARAG Mock API")

@app.get("/health")
def health():
    return {"status": "ok"}