"""End-to-end regression check via HTTP — exercises full app pipeline."""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

CASES = [
    ("bohkuraa", "ބޮއްކުރާ"),
    ("kuru",     "ކުރު"),
    ("karu",     "ކަރު"),
    ("bas",      "ބަސް"),
    ("Aharennakee Dhivehi bahun vaahaka dhakkaa meeheh", None),
    ("Maadhama haveeruge bahdhaluvun cancel kohffi",     None),
]


def call(text):
    req = urllib.request.Request(
        "http://localhost:5001/transliterate",
        data=json.dumps({"text": text, "direction": "latin2thaana"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    final = None
    with urllib.request.urlopen(req, timeout=30) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if line.startswith("data: "):
                d = json.loads(line[6:])
                if not d.get("partial", True):
                    final = d.get("thaana")
    return final


for latin, expected in CASES:
    out = call(latin)
    if expected is None:
        print(f"{latin!r}\n  -> {out!r}\n")
    else:
        flag = "PASS" if out == expected else "FAIL"
        print(f"[{flag}] {latin!r}\n  got:      {out!r}\n  expected: {expected!r}\n")
