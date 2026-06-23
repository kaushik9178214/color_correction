from fashn_human_parser import FashnHumanParser
import cv2
import os

parser = FashnHumanParser()

ref_file = None

for f in os.listdir("input"):
    if f.startswith("_color-ref"):
        ref_file = os.path.join("input", f)
        break

print("Using:", ref_file)

img = cv2.imread(ref_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = parser.predict(img)

print(type(result))

try:
    print("Shape:", result.shape)
except:
    pass

try:
    print("Dtype:", result.dtype)
except:
    pass

print(result)