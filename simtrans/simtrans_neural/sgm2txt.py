import sys
import re

TAG_RE = re.compile(r'<[^>]+>')

for line in sys.stdin:
    if line.startswith('<doc') or line.startswith('</doc>'):
        continue
    sys.stdout.write(TAG_RE.sub('', line.lower()))
