# ai-back-end

FastAPI web service.

## Run

```bash
uv run uvicorn main:server --reload
```

Server starts at `http://localhost:8000`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/messages` | Returns list of messages |

API docs available at `http://localhost:8000/docs`.
