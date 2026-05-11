import re
with open('ds4fa/ds4_hip.cpp', 'r') as f:
    text = f.read()

text = text.replace('fprintf(stderr, "ds4_hip: 128GB Strix Halo APU detected. Defaulting to safe memory profiles.\n");', 'fprintf(stderr, "ds4_hip: 128GB Strix Halo APU detected. Defaulting to safe memory profiles.\\n");')

with open('ds4fa/ds4_hip.cpp', 'w') as f:
    f.write(text)
