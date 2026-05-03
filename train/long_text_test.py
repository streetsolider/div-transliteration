"""Reproduce the long-text input through HTTP to diagnose the network error."""
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

LONG_TEXT = """Target bunee bunebalaashey rajje ga ulhey women rights activist eh. Thankolheh varugadha ehcheh. Kurusee ah dhey komme ehcheh eba jaha ey. Nubala ey thedheh dhogeh ves. International media ah meethi varugadhayah jassanshey miulhenee. Adhadhu meehunney aslu amillayah beynunee. E anhen gola meehakaa rattehi viyas anekaku rulhi aadhey kamah. Hama msg kohfa eba huttey eynaya dhuru vaasho kiyaafa. Meenayakah angain ves bunenuleve ey. Fahathuga polihun baithihbaafa ey thibeny. Inna hisaabah gossa ey othy ves. Msg thakuga eba huttey.

Ekolhun bunee eyrun vaane dho ey verikamehga hure furathama meeha ah. Fiyavalhu naalhaa madun huri nama mi hindhaalevunees ey.

Target bunee dhehki vaahaka akashey mikan vaany ves. Ithuru ehves kameh nuvaanenun hey. Ibu ves meege ehcheh dhiya ehnun hey. Balaabalaashey activist eh.

Ekolhun bunee mihaaru dhn noolheynun hey. Dhn ulheny kalhu manje ey. Amillayah promote vaa meehekey ei.

Target bunee shameem kurimahchah leeves aslu beyrah dhahkaashey. Adhi ulhey ey ithuru dhemeehun. Meena dheke ey emmen loabivany. Mi emmen thibeny eh gothakashey. Eynage kamakee meena ah faaralumey. Hure ey device harukohfa thanthaaga ves. Dhehkeyne ehchehi eba huttey. Ehnve ey activist ekey mi hoadhany ekan kuran. Mihuriha dhuvahu aharemen thibee ey alhaanulaa. Dhn mi fetty ey. Ingeyhey vrh galhi gola ekey ei. Mainbafain dharin vikkaigen bodu kohfa ey thibee. Eyna dhari akee ves ekahala kommeves meehegge dhari ekey. Eyna bappa akee drug addict ekey. Eyna mamma akee jangiya nazim ge vagu bitey. Dhn visnaalaashey mulhi ethi."""

word_count = len(LONG_TEXT.split())
print(f"Input: {word_count} words, {len(LONG_TEXT.encode('utf-8'))} bytes")

req = urllib.request.Request(
    "http://localhost:5001/transliterate",
    data=json.dumps({"text": LONG_TEXT, "direction": "latin2thaana"}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)

t0 = time.perf_counter()
last_event = t0
event_count = 0
final = None
max_gap = 0.0
print("\nStreaming events:")
with urllib.request.urlopen(req, timeout=180) as r:
    for raw in r:
        line = raw.decode("utf-8").strip()
        if line.startswith("data: "):
            now = time.perf_counter()
            gap = now - last_event
            if gap > max_gap:
                max_gap = gap
            last_event = now
            event_count += 1
            d = json.loads(line[6:])
            status = d.get("status", "")
            progress = d.get("progress", "?")
            partial = d.get("partial", None)
            print(f"  +{now-t0:>6.2f}s  gap={gap:>5.2f}s  progress={progress}  partial={partial}  status={status[:60]!r}")
            if partial is False:
                final = d.get("thaana")

elapsed = time.perf_counter() - t0
print(f"\n=== Done ===")
print(f"  total time:     {elapsed:.2f}s")
print(f"  events received: {event_count}")
print(f"  longest silent gap between events: {max_gap:.2f}s")
print(f"  final thaana length: {len(final) if final else 'None'} chars")
