import os, time, re, html, json, requests, argparse
from urllib.parse import urlencode
from dotenv import load_dotenv
from tqdm import tqdm


from dotenv import load_dotenv
load_dotenv()


TMDB_API_TOKEN = os.environ.get("TMDB_API_TOKEN")
BASE = "https://api.themoviedb.org/3"


def tmdb(path, **params):
    """Call TMDB API with given path and parameters."""
    if not TMDB_API_TOKEN:
        raise RuntimeError("TMDB_API_TOKEN is not set. Put it in your environment or .env file.")
    # Drop unset/empty query params so we don't send e.g. with_genres=None
    safe_params = {k: v for k, v in params.items() if v is not None and v != ""}
    url = f"{BASE}{path}?{urlencode(safe_params)}"
    headers = {"Authorization": f"Bearer {TMDB_API_TOKEN}"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def clean(s):
    if not s:
        return ""
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    return re.sub(r"\s+", " ", s).strip()


def discover_movies(pages=10, **kw):
    """Fetch popular movies via /discover/movie."""
    out = []
    for p in tqdm(range(1, pages + 1), total=pages, desc="Discover pages", unit="page"):
        data = tmdb("/discover/movie", page=p, **kw)
        out += data.get("results", [])
        time.sleep(0.2)
    return out


def enrich_movie(mid):
    """Pull detailed movie info, credits, keywords, and reviews."""
    d = tmdb(f"/movie/{mid}")
    rev = tmdb(f"/movie/{mid}/reviews").get("results", [])[:5]
    kw = tmdb(f"/movie/{mid}/keywords").get("keywords", [])
    cred = tmdb(f"/movie/{mid}/credits")
    dirs = [p["name"] for p in cred.get("crew", []) if p.get("job") == "Director"][:3]
    cast = [p["name"] for p in cred.get("cast", [])][:5]
    keywords = [k["name"] for k in kw][:20]

    title = d.get("title")
    tagline = clean(d.get("tagline"))
    overview = clean(d.get("overview"))
    reviews = " ".join(clean(x.get("content")) for x in rev)[:2000]

    idx_text = " — ".join(filter(None, [title, tagline])) + ". " + overview
    if keywords:
        idx_text += " Keywords: " + "; ".join(keywords) + "."
    if reviews:
        idx_text += " Reviews: " + reviews

    # Construct full poster URL
    poster_path = d.get("poster_path")
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

    doc = {
        "id": f"tmdb:movie:{d['id']}",
        "title": title,
        "year": int(d["release_date"][:4]) if d.get("release_date") else None,
        "genres": [g["name"] for g in d.get("genres", [])],
        "keywords": keywords,
        "people": {"director": dirs, "cast": cast},
        "language": d.get("original_language"),
        "runtime": d.get("runtime"),
        "tmdb_url": f"https://www.themoviedb.org/movie/{d['id']}",
        "poster_url": poster_url,
        "index_text": idx_text
    }
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=200, help="How many /discover pages to fetch")
    ap.add_argument("--out", default="movies_docs.json", help="Output path (default: project root)")
    ap.add_argument("--sort_by", default="vote_count.desc", help="TMDB discover sort_by")
    ap.add_argument("--language", default="en-US", help="TMDB language")
    ap.add_argument("--with_genres", default=None, help="Comma-separated TMDB genre ids (optional)")
    args = ap.parse_args()

    # Step 1: discover basic list of movies
    seed = discover_movies(
        pages=args.pages,
        sort_by=args.sort_by,
        language=args.language,
        with_genres=args.with_genres
    )
    movie_ids = [m["id"] for m in seed]

    # Step 2: enrich each movie
    docs = []
    for mid in tqdm(movie_ids, desc="Enrich movies", unit="movie"):
        try:
            docs.append(enrich_movie(mid))
            time.sleep(0.25)  # Be kind to the API
        except Exception as e:
            print("skip", mid, e)

    # Step 3: save to JSON
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print("Collected:", len(docs), "→", args.out)


if __name__ == "__main__":
    main()
