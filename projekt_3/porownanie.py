"""Generuje ogłoszenia dla zdjęć z przyklady/: każdy model bez tytułu, a model domyślny także z tytułem.
Wyniki zapisuje do wyniki/wyniki.json i wyniki/porownanie.md."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from utils import DEFAULT_PROMPT, build_prompt, generate_listing

MODELS = ["gpt-6-luna", "gpt-6.1-terra", "gpt-6.2-sol", "gpt-4.1-mini"]
TITLE_MODEL = "gpt-6-luna"
TITLES = {
    "1_salon.jpg": "Mieszkanie 2 pokoje 48 m2 z balkonem, Poznań Jeżyce",
    "2_kuchnia.jpg": "Mieszkanie 3-pokojowe 62 m2 po remoncie, Poznań Wilda",
    "3_golf.jpg": "Volkswagen Golf IV 1.4 16V benzyna 1998",
    "4_sofa.jpg": "Sofa IKEA Klippan 2-osobowa",
    "5_motocykl.jpg": "Honda XR 250 Baja 2004",
}

client = OpenAI()
images = sorted(Path("przyklady").glob("*.jpg"))
jobs = [(image, model, "") for image in images for model in MODELS]
jobs += [(image, TITLE_MODEL, TITLES[image.name]) for image in images]


def run(job):
    image, model, title = job
    result = generate_listing(client, image.read_bytes(), build_prompt(DEFAULT_PROMPT, title), model)
    return {"zdjecie": image.name, "model": model, "tytul_wejsciowy": title, **result}


with ThreadPoolExecutor(max_workers=5) as pool:
    results = list(pool.map(run, jobs))

Path("wyniki").mkdir(exist_ok=True)
Path("wyniki/wyniki.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

lines = ["| Zdjęcie | Model | Tytuł wejściowy | Wygenerowany tytuł | Parametry | Do uzupełnienia | Czas [s] | Tokeny we/wy |",
         "|---|---|---|---|---|---|---|---|"]
for r in results:
    listing = r["listing"]
    params = "; ".join(f"{k}: {v}" for k, v in listing.get("parametry", {}).items())
    lines.append(f"| {r['zdjecie']} | {r['model']} | {r['tytul_wejsciowy'] or '-'} | {listing.get('tytul', '')} | {params} | "
                 f"{len(listing.get('do_uzupelnienia', []))} | {r['seconds']} | {r['input_tokens']}/{r['output_tokens']} |")
Path("wyniki/porownanie.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
