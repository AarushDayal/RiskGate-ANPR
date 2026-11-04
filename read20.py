import glob, os, sys, csv
import xml.etree.ElementTree as ET
import cv2, easyocr

root, n = sys.argv[1], int(sys.argv[2])  # ./datacluster 20
os.makedirs("crops", exist_ok=True)
reader = easyocr.Reader(["en"], gpu=False)  # GPU ho to True kar
rows = []

for x in sorted(glob.glob(f"{root}/Annotations/*.xml"))[:n]:
    t = ET.parse(x).getroot()
    name = t.findtext("filename")
    img = cv2.imread(f"{root}/images/{name}")
    if img is None:
        print("missing image", name)
        continue
    o = t.find("object")
    bb = o.find("bndbox")
    x1, y1, x2, y2 = [
        int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")
    ]
    h, w = img.shape[:2]
    crop = img[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
    cv2.imwrite(f"crops/{name}", crop)

    res = reader.readtext(crop)  # [(bbox, text, conf), ...]
    text = " ".join(r[1] for r in res)
    conf = round(sum(r[2] for r in res) / len(res), 3) if res else 0.0
    print(f"{name} | easyocr='{text}' | conf={conf}")
    rows.append([name, text, conf, ""])

with open("labels.csv", "w", newline="") as f:
    csv.writer(f).writerows([["image", "easyocr", "conf", "truth"]] + rows)
