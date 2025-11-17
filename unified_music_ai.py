import base64, json, os
from openai import OpenAI

MODEL = os.getenv("MODEL", "gpt-5-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = (
  "You are a concise multimedia tagger for music discovery. "
  "Given an image and/or text, return STRICT JSON with keys: "
  "description (str), keywords (list[str]), moods (list[str]), "
  "genre (str), search_queries (list[str]). "
  "Respond with exactly one genre, not a list."
)

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        ext = path.split(".")[-1].lower()
        mime = "png" if ext == "png" else "jpeg" if ext in {"jpg", "jpeg"} else ext
        return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()

def call_model(parts):
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": parts},
        ],
    )
    return json.loads(resp.choices[0].message.content)

def analyze(image_path=None, text=None):
    if not image_path and not text:
        raise SystemExit("Provide --image and/or --text")

    content_parts = []
    prompt_bits = []
    if image_path: prompt_bits.append("Use the image to infer scene, instruments, style, and vibe.")
    if text:       prompt_bits.append("Also use the text as hints about intent or keywords.")
    content_parts.append({"type": "text", "text": " ".join(prompt_bits)})

    if image_path:
        data_url = encode_image(image_path)
        content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
    if text:
        content_parts.append({"type": "text", "text": f"TEXT: {text}"})

    out = call_model(content_parts)

    # post-process search queries: make unique & short
    seen, uniq = set(), []
    for q in out.get("search_queries", []):
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            uniq.append(q)
    out["search_queries"] = uniq[:6]

    return out

if __name__ == "__main__":
    import argparse, sys
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Missing OPENAI_API_KEY. Set it in your env or .env.")
    p = argparse.ArgumentParser()
    p.add_argument("--image", help="path to image (optional)")
    p.add_argument("--text", help="free text or keywords (optional)")
    args = p.parse_args()
    result = analyze(args.image, args.text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
