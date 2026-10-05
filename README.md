# Portfolio RAG Assistant (FastAPI + LangChain + React)

A chat widget for your portfolio that answers questions about you, grounded in your own content.

## 1. Run the backend locally

```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                              # then add your GOOGLE_API_KEY
```

1. Replace the sample files in `backend/data/` with real content (resume, one file per project, skills, contact). Use `## Headings` to separate topics.
2. Build the index: `python ingest.py` (re-run whenever you edit `data/`).
3. Start the API: `uvicorn main:app --reload`
4. Test it at http://localhost:8000/docs

## 2. Add the widget to your React portfolio

1. Copy `frontend/ChatWidget.jsx` and `ChatWidget.css` into `src/components/`.
2. Set `API_URL` at the top of `ChatWidget.jsx`.
3. Render `<ChatWidget />` once, e.g. at the bottom of `App.jsx`.
4. Adjust the CSS variables at the top of `ChatWidget.css` to match your site's colors and font.

## 3. Deploy

- **Backend:** Render or Railway. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
  Commit the `index/` folder, and set `GOOGLE_API_KEY` and `ALLOWED_ORIGINS` (your Vercel URL) as environment variables.
- **Frontend:** push the portfolio to Vercel as usual after changing `API_URL`.

## Resume bullet

Built and deployed a RAG assistant for my portfolio using FastAPI, LangChain, FAISS and Gemini, with streaming (SSE) responses, source attribution, per-IP rate limiting and a React chat widget.
