import re
with open('ds4fa/ds4.c', 'r') as f:
    text = f.read()

match = re.search(r'uint64_t ds4_context_memory_estimate\(.*?\)(.*?)\n\}', text, re.DOTALL)
if match:
    print(match.group(0))
