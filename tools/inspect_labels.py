from fashn_human_parser import FashnHumanParser
import cv2
import numpy as np
import os

parser = FashnHumanParser()

ref_file = None

for f in os.listdir("input"):
    if f.startswith("_color-ref"):
        ref_file = os.path.join("input", f)
        break

img = cv2.imread(ref_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = parser.predict(img)

unique_labels = np.unique(mask)

print("Unique labels:")
print(unique_labels)

print("\nLabel names:")

for label in unique_labels:
    try:
        print(label, "->", parser.get_label_name(int(label)))
    except Exception as e:
        print(label, "-> UNKNOWN")