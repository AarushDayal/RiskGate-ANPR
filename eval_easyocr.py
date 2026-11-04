import glob, re, sys, csv
import xml.etree.ElementTree as ET
import cv2, easyocr

root = sys.argv[1]  # ./datacluster
min_h = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # chhote crops skip karne ke liye

reader = easyocr.Reader(["en"], gpu=False)


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def cer(a, b):  # edit distance / len(truth)
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1] / max(len(a), 1)


IND = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$|^\d{2}BH\d{4}[A-Z]{2}$")

rows = []
for x in sorted(glob.glob(f"{root}/Annotations/*.xml")):
    t = ET.parse(x).getroot()
    name = t.findtext("filename")
    img = cv2.imread(f"{root}/images/{name}")
    if img is None:
        continue
    H, W = img.shape[:2]
    for k, o in enumerate(t.findall("object")):
        truth = None
        for a in o.iter("attribute"):
            if a.findtext("name") == "number_plate_text":
                truth = a.findtext("value")
        if not truth:
            continue
        bb = o.find("bndbox")
        x1, y1, x2, y2 = [
            int(float(bb.findtext(c))) for c in ("xmin", "ymin", "xmax", "ymax")
        ]
        crop = img[max(0, y1) : min(H, y2), max(0, x1) : min(W, x2)]
        if crop.size == 0 or crop.shape[0] < min_h:
            continue
        res = reader.readtext(crop)
        pred = norm("".join(r[1] for r in res))
        conf = sum(r[2] for r in res) / len(res) if res else 0.0
        tr = norm(truth)
        rows.append(
            dict(
                img=name,
                obj=k,
                h=crop.shape[0],
                truth=tr,
                pred=pred,
                conf=round(conf, 3),
                ok=int(pred == tr),
                cer=round(cer(pred, tr), 3),
                valid=int(bool(IND.match(pred))),
            )
        )
        print(rows[-1])

with open("eval_easyocr.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

n = len(rows)
good = [r for r in rows if r["ok"]]
bad = [r for r in rows if not r["ok"]]
avg = lambda L, k: round(sum(r[k] for r in L) / len(L), 3) if L else None
print(f"\nplates={n}  exact_match={len(good)}/{n}  mean_CER={avg(rows, 'cer')}")
print(f"mean conf  correct={avg(good, 'conf')}  wrong={avg(bad, 'conf')}")
print(f"grammar-valid: correct={avg(good, 'valid')}  wrong={avg(bad, 'valid')}")
