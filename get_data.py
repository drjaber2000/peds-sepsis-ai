"""Download the data used by the notebooks (not redistributed in this repository).

1. 2024 Pediatric Sepsis Challenge synthetic training set (Borealis, doi:10.5683/SP3/TFAV36,
   CC BY-NC-SA 4.0) -> PSDC_SyntheticTrainingData_Dataset_ODR.tab + data dictionary (.docx),
   and data_dictionary.txt extracted from the .docx.
2. WHO Child Growth Standards LMS tables from the WHO `anthro` R package (GPL-3) -> who_growth/

Run once from the repository root:  python get_data.py
"""
import hashlib
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

BOREALIS = "https://borealisdata.ca/api/access/datafile/"
FILES = {  # file name: (Borealis file id, md5)
    "PSDC_SyntheticTrainingData_Dataset_ODR.tab": (966789, "256ea65a82c974c32c0e53c9423ec194"),
    "PSDC_SyntheticTrainingData_DataDictionary_ODR.docx": (966788, "7c661c5087361006ce22eef1421c3a91"),
}
WHO = ("https://raw.githubusercontent.com/WorldHealthOrganization/anthro/master/"
       "data-raw/growthstandards/{}.txt")
WHO_TABLES = ["weianthro", "lenanthro", "wflanthro", "wfhanthro", "acanthro"]


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def fetch(url, dest):
    if Path(dest).exists():
        print(f"  exists  {dest}")
        return
    print(f"  get     {dest}")
    urllib.request.urlretrieve(url, dest)


print("Pediatric Sepsis Challenge data (Borealis):")
for name, (file_id, checksum) in FILES.items():
    fetch(BOREALIS + str(file_id), name)
    assert md5(name) == checksum, f"{name}: checksum mismatch - the published file may have changed"

print("Data dictionary -> data_dictionary.txt")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
root = ET.fromstring(zipfile.ZipFile(list(FILES)[1]).read("word/document.xml"))
rows = [" | ".join("".join(t.text or "" for t in cell.iter(W + "t")).strip() for cell in tr.findall(W + "tc"))
        for tbl in root.iter(W + "tbl") for tr in tbl.iter(W + "tr")]
Path("data_dictionary.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

print("WHO growth standards (anthro):")
Path("who_growth").mkdir(exist_ok=True)
for t in WHO_TABLES:
    fetch(WHO.format(t), f"who_growth/{t}.txt")

print("Done.")
