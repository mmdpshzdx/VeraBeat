import base64, json, os
from openai import OpenAI

MODEL = os.getenv("MODEL", "gpt-5-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = (
  "You are a music genre classifier. "
  "Given ONLY an image, return STRICT JSON with the key: genre (str). "
  "GENRE MUST BE EXACTLY ONE WORD OR ONE PHRASE. "
  "DO NOT RETURN ANYTHING ELSE. "
  "NO descriptions, NO keywords, NO moods, NO lists — ONLY genre."
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

def analyze(image_path):
    if not image_path:
        raise SystemExit("Provide an image path.")

    content = [
        {"type": "text", "text": "Classify the music genre most associated with this image."}
    ]

    data_url = encode_image(image_path)
    content.append({"type": "image_url", "image_url": {"url": data_url}})

    out = call_model(content)

    # Force single genre output
    genre = out.get("genre")
    if isinstance(genre, list):
        out["genre"] = genre[0] if genre else "unknown"
    elif isinstance(genre, str):
        out["genre"] = genre.split(",")[0].split("/")[0].strip()
    else:
        out["genre"] = "unknown"

    return out

if __name__ == "__main__":
    import argparse, sys
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Missing OPENAI_API_KEY. Set it in your env or .env.")

    p = argparse.ArgumentParser()
    p.add_argument("image", help="path to image")
    args = p.parse_args()

    result = analyze(args.image)
    print(json.dumps(result, indent=2, ensure_ascii=False))
