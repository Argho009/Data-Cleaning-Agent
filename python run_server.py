import uvicorn

if __name__ == "__main__":
    # Use 127.0.0.1 so logs and browser URL match (localhost only).
    uvicorn.run("app:app", host="127.0.0.1", port=7860)