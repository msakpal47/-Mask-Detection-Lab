import argparse
import os
import cv2
import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", required=True, help="search query")
    ap.add_argument("-o", "--output", required=True, help="output directory")
    args = ap.parse_args()

    api_key = os.getenv("BING_API_KEY")
    if not api_key:
        raise RuntimeError("Set BING_API_KEY environment variable to your Bing Image Search API key")

    max_results = 500
    group_size = 50
    url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
    exceptions = (IOError, FileNotFoundError, requests.RequestException, requests.HTTPError, requests.ConnectionError, requests.Timeout)

    term = args.query
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": term, "offset": 0, "count": group_size}
    s = requests.Session()
    s.headers.update(headers)

    r = s.get(url, params=params, timeout=30)
    r.raise_for_status()
    results = r.json()
    est_num = min(results.get("totalEstimatedMatches", 0), max_results)
    total = 0

    for offset in range(0, est_num, group_size):
        params["offset"] = offset
        r = s.get(url, params=params, timeout=30)
        r.raise_for_status()
        results = r.json()
        for v in results.get("value", []):
            try:
                content_url = v.get("contentUrl")
                if not content_url:
                    continue
                resp = s.get(content_url, timeout=30)
                ext = os.path.splitext(content_url)[1]
                if not ext or len(ext) > 5:
                    ext = ".jpg"
                p = os.path.join(out_dir, f"{str(total).zfill(8)}{ext}")
                with open(p, "wb") as f:
                    f.write(resp.content)
            except exceptions:
                if 'p' in locals() and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                continue
            image = cv2.imread(p)
            if image is None:
                try:
                    os.remove(p)
                except Exception:
                    pass
                continue
            total += 1

if __name__ == "__main__":
    main()

