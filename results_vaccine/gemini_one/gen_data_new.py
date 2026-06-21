"""
parse_results.py
────────────────
Parsuje pliki wyników o nazwach pasujących do jednego z dwóch wzorców:
  Wariant A: results_{KORPUS}_{strategy}_{k}_{wariant}_{model}_{typ}.txt
             np. results_POLITYCZNE_dif_3_ONE_gemini_dva_metrics.txt
  Wariant B: results_{KORPUS}_{k}_{STRATEGY}_{wariant}_{model}_{typ}.txt
             np. results_ZDROWOTNE_3_DIF_ONE_gemini_dva_metrics.txt

Generuje słownik w formacie:
  model → metric (f1/cosine/spearman) → strategy (sim/dif) → stat (mu/sd) → domain → [k=3, k=5, k=10]

Wymagania: standardowa biblioteka Python 3.8+
"""

import re
import json
from pathlib import Path

# ─── konfiguracja ──────────────────────────────────────────────────────────────

INPUT_DIR   = Path(".")
OUTPUT_FILE = Path("./metrics_output.json")
OUTPUT_PY   = Path("./metrics_output.py")

K_ORDER = [3, 5, 10]

# Wariant A: {corpus}_{strategy}_{k}_{variant}_{model}_{typ}  (np. POLITYCZNE_dif_3_ONE_gemini)
_PAT_A = re.compile(
    r"results_(?P<corpus>[^_]+)_"
    r"(?P<strategy>sim|dif)_"
    r"(?P<k>\d+)_"
    r"(?P<variant>[^_]+)_"
    r"(?P<model>[^_]+)_"
    r"(?P<filetype>.+)\.txt$",
    re.IGNORECASE,
)

# Wariant B: {corpus}_{k}_{STRATEGY}_{variant}_{model}_{typ}  (np. ZDROWOTNE_3_DIF_ONE_gemini)
_PAT_B = re.compile(
    r"results_(?P<corpus>[^_]+)_"
    r"(?P<k>\d+)_"
    r"(?P<strategy>sim|dif)_"
    r"(?P<variant>[^_]+)_"
    r"(?P<model>[^_]+)_"
    r"(?P<filetype>.+)\.txt$",
    re.IGNORECASE,
)

def match_filename(name: str):
    """Próbuje obu wzorców; zwraca (strategy, k, model, filetype) lub None."""
    for pat in (_PAT_A, _PAT_B):
        m = pat.match(name)
        if m:
            return (
                m.group("strategy").lower(),
                int(m.group("k")),
                m.group("model").lower(),
                m.group("filetype").lower(),
            )
    return None

# ─── parsery treści ────────────────────────────────────────────────────────────

def parse_classification(text: str) -> dict:
    """
    Zwraca {domain: {accuracy, precision, recall, f1}}.

    Format wejścia:
        GOVERNMENT_INTERVENTION:
          Dokładność: 0.9072
          Wynik F1:   0.9071
    """
    label_map = {
        "Dokładność": "accuracy",
        "Precyzja":   "precision",
        "Czułość":    "recall",
        "Wynik F1":   "f1",
    }
    results = {}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^[A-Z_]+:$", line):          # nagłówek domeny
            current = line.rstrip(":").lower()
            results[current] = {}
        elif current and ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            if key in label_map:
                try:
                    results[current][label_map[key]] = float(val.strip())
                except ValueError:
                    pass
    return results


def parse_dva(text: str) -> dict:
    """
    Zwraca {domain: {cosine: {mu, sd}, proj_diff: {mu, sd}, spearman: {mu, sd}}}.

    Format wiersza danych:
        government_intervention | 9 | 0.866±0.024 | 0.515±0.048 | 0.675±0.053
        Kolumny: Domena | N par | Cos | Proj.Diff | Spearman [| Pearson]
    """
    def split(cell: str):
        m = re.match(r"([\d.]+)±([\d.]+)", cell.strip())
        return (float(m.group(1)), float(m.group(2))) if m else (None, None)

    results = {}
    for line in text.splitlines():
        if "|" not in line or "Domena" in line or line.strip().startswith("-"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        domain = parts[0].lower()
        cos_mu,  cos_sd  = split(parts[2])
        pdif_mu, pdif_sd = split(parts[3])
        spe_mu,  spe_sd  = split(parts[4])
        results[domain] = {
            "cosine":    {"mu": cos_mu,  "sd": cos_sd},
            "proj_diff": {"mu": pdif_mu, "sd": pdif_sd},
            "spearman":  {"mu": spe_mu,  "sd": spe_sd},
        }
    return results


# ─── akumulator ───────────────────────────────────────────────────────────────
# Struktura: raw[model][metric][strategy][stat][domain][k] = float

def set_val(raw, model, metric, strategy, stat, domain, k, value):
    raw \
        .setdefault(model, {}) \
        .setdefault(metric, {}) \
        .setdefault(strategy, {}) \
        .setdefault(stat, {}) \
        .setdefault(domain, {})[k] = value


# ─── główna logika ─────────────────────────────────────────────────────────────

def main():
    files = sorted(INPUT_DIR.glob("*.txt"))
    if not files:
        print(f"[BŁĄD] Brak plików .txt w: {INPUT_DIR}")
        return

    raw = {}
    processed, skipped = [], []

    for filepath in files:
        parsed = match_filename(filepath.name)
        if parsed is None:
            skipped.append(filepath.name)
            continue

        strategy, k, model, filetype = parsed
        processed.append((filepath.name, strategy, k, model, filetype))

    # ─── raport parsowania nazw ────────────────────────────────────────────────
    width = max((len(n) for n, *_ in processed), default=20) + 2

    print(f"{'═'*80}")
    print(f"  PRZETWORZONE: {len(processed)}  │  POMINIĘTE: {len(skipped)}")
    print(f"{'═'*80}")
    print(f"  {'PLIK':<{width}}  STRATEGY   K    MODEL     TYP")
    print(f"  {'-'*width}  ---------  ---  --------  -------------------------")
    for name, strat, k_val, mdl, ft in processed:
        print(f"  {name:<{width}}  {strat:<9}  {k_val:<3}  {mdl:<8}  {ft}")
    if skipped:
        print()
        print("  POMINIĘTE (brak dopasowania do wzorca nazwy):")
        for name in skipped:
            print(f"    ✗  {name}")
    print(f"{'═'*80}\n")

    # ─── właściwe przetwarzanie ────────────────────────────────────────────────
    for filepath in files:
        parsed = match_filename(filepath.name)
        if parsed is None:
            print(f"[POMINIĘTO] {filepath.name}")
            continue

        strategy, k, model, filetype = parsed
        text     = filepath.read_text(encoding="utf-8")

        if "classification_metrics_results" in filetype:
            for domain, vals in parse_classification(text).items():
                if "f1" in vals:
                    set_val(raw, model, "f1", strategy, "mu", domain, k, vals["f1"])

        elif "dva_metrics" in filetype:
            for domain, vals in parse_dva(text).items():
                for metric in ("cosine", "spearman"):
                    for stat in ("mu", "sd"):
                        v = vals[metric][stat]
                        if v is not None:
                            set_val(raw, model, metric, strategy, stat, domain, k, v)
        else:
            print(f"[POMINIĘTO] nieznany typ: {filepath.name}")

    # ─── budowanie finalnej struktury ─────────────────────────────────────────
    # raw[model][metric][strategy][stat][domain] = {3: v, 5: v, 10: v}
    # → output[model][metric][strategy][stat][domain] = [v3, v5, v10]

    output = {}
    for model, metrics in raw.items():
        output[model] = {}
        for metric, strategies in metrics.items():
            output[model][metric] = {}
            for strategy, stats in strategies.items():
                output[model][metric][strategy] = {}
                for stat, domains in stats.items():
                    output[model][metric][strategy][stat] = {}
                    for domain, k_vals in domains.items():
                        output[model][metric][strategy][stat][domain] = [
                            k_vals.get(k) for k in K_ORDER
                        ]

    # ─── zapis JSON ───────────────────────────────────────────────────────────

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"[OK] JSON  → {OUTPUT_FILE}")

    # ─── zapis Python ─────────────────────────────────────────────────────────

    def py_repr(obj, depth=0):
        pad  = "    " * depth
        pad1 = "    " * (depth + 1)
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            items = [f"{pad1}{repr(key)}: {py_repr(val, depth+1)}" for key, val in obj.items()]
            return "{\n" + ",\n".join(items) + ",\n" + pad + "}"
        if isinstance(obj, list):
            parts = [(f"{v:.4f}" if isinstance(v, float) else str(v)) for v in obj]
            return "[" + ", ".join(parts) + "]"
        if isinstance(obj, float):
            return f"{obj:.4f}"
        return repr(obj)

    lines = [f"k = {K_ORDER}", ""]
    for model, data in output.items():
        lines.append(f"{model} = {py_repr(data)}")
        lines.append("")

    with open(OUTPUT_PY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Python → {OUTPUT_PY}")

    # ─── podgląd ──────────────────────────────────────────────────────────────
    print("\n=== PODGLĄD ===\n")
    print(json.dumps(output, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()

