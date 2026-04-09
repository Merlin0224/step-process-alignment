import time
from pathlib import Path
import requests

URL = "https://huggingface.co/Qwen/Qwen2.5-3B/resolve/main/model-00001-of-00002.safetensors?download=true"
OUT = Path("/root/autodl-tmp/models/Qwen/Qwen2.5-3B/model-00001-of-00002.safetensors")
TMP = OUT.with_suffix(OUT.suffix + ".part")
OUT.parent.mkdir(parents=True, exist_ok=True)

attempt = 0
while True:
    attempt += 1
    written = TMP.stat().st_size if TMP.exists() else 0
    headers = {"Range": f"bytes={written}-"} if written > 0 else {}
    print(f"ATTEMPT {attempt} written={written}", flush=True)
    try:
        with requests.get(URL, stream=True, allow_redirects=True, headers=headers, timeout=(30, 300)) as r:
            print(f"STATUS {r.status_code}", flush=True)
            print(f"FINAL_URL {r.url}", flush=True)
            if r.status_code == 503 and written > 0:
                print("RANGE_FAILED_RESTART_FROM_ZERO", flush=True)
                TMP.unlink(missing_ok=True)
                time.sleep(3)
                continue
            if r.status_code not in (200, 206):
                raise RuntimeError(f"unexpected status: {r.status_code}")
            if r.status_code == 200 and written > 0:
                print("SERVER_IGNORED_RANGE restarting_from_zero", flush=True)
                written = 0
                TMP.unlink(missing_ok=True)
            total_header = r.headers.get("content-length")
            total = int(total_header) + written if total_header is not None else None
            print(f"TOTAL {total}", flush=True)
            chunk_size = 8 * 1024 * 1024
            last_report = time.time()
            last_bytes = written
            mode = "ab" if written > 0 else "wb"
            with open(TMP, mode) as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    now = time.time()
                    if now - last_report >= 10:
                        speed = (written - last_bytes) / max(now - last_report, 1e-6) / (1024 ** 2)
                        if total:
                            pct = written / total * 100
                            print(f"PROGRESS bytes={written} total={total} pct={pct:.2f} speed_mb_s={speed:.2f}", flush=True)
                        else:
                            print(f"PROGRESS bytes={written} speed_mb_s={speed:.2f}", flush=True)
                        last_report = now
                        last_bytes = written
            TMP.replace(OUT)
            print(f"DONE {OUT}", flush=True)
            break
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {e}", flush=True)
        time.sleep(min(30, 3 * attempt))
