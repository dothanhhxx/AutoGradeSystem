"""Dump theme colors/fonts and key layout text styles of a template."""
import re
import sys

from lxml import etree
from pptx import Presentation

p = Presentation(sys.argv[1])
m = p.slide_masters[0]
th = [r for r in m.part.rels.values() if "theme" in r.reltype][0].target_part.blob.decode("utf8")
print("COLORS", re.findall(r"<a:(dk1|lt1|dk2|lt2|accent\d|hlink)>(.*?)</a:\1>", th))
print("FONTS", re.findall(r'<a:(?:major|minor)Font>\s*<a:latin typeface="([^"]*)"', th))
for i in (0, 1, 3, 5, 7):
    layout = m.slide_layouts[i]
    print("-----", i, layout.name)
    for ph in layout.placeholders:
        if ph.placeholder_format.idx in (0, 1, 2):
            print(ph.name, etree.tostring(ph._element.txBody, pretty_print=True).decode()[:1000])
print("MASTER titleStyle")
print(etree.tostring(m._element.find(".//{*}titleStyle"), pretty_print=True).decode()[:1500])
print("MASTER bodyStyle")
print(etree.tostring(m._element.find(".//{*}bodyStyle"), pretty_print=True).decode()[:2500])
