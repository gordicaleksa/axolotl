from copy import deepcopy
import os
import re
import string
import warnings
from collections import defaultdict, OrderedDict, Counter
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import concurrent
re_attr_val = re.compile(r' (.+?)="(.*?)"')

from tqdm import tqdm

from json import load

from conllu import parse_incr
from datasets import load_dataset



# This dictionary is a mapping between Serbian Cyrillic and Latin.
# Found it here: https://github.com/opendatakosovo/cyrillic-transliteration/blob/master/cyrtranslit/mapping.py
SR_CYR_TO_LAT_DICT = {
    u'А': u'A', u'а': u'a',
    u'Б': u'B', u'б': u'b',
    u'В': u'V', u'в': u'v',
    u'Г': u'G', u'г': u'g',
    u'Д': u'D', u'д': u'd',
    u'Ђ': u'Đ', u'ђ': u'đ',
    u'Е': u'E', u'е': u'e',
    u'Ж': u'Ž', u'ж': u'ž',
    u'З': u'Z', u'з': u'z',
    u'И': u'I', u'и': u'i',
    u'Ј': u'J', u'ј': u'j',
    u'К': u'K', u'к': u'k',
    u'Л': u'L', u'л': u'l',
    u'Љ': u'Lj', u'љ': u'lj',
    u'М': u'M', u'м': u'm',
    u'Н': u'N', u'н': u'n',
    u'Њ': u'Nj', u'њ': u'nj',
    u'О': u'O', u'о': u'o',
    u'П': u'P', u'п': u'p',
    u'Р': u'R', u'р': u'r',
    u'С': u'S', u'с': u's',
    u'Т': u'T', u'т': u't',
    u'Ћ': u'Ć', u'ћ': u'ć',
    u'У': u'U', u'у': u'u',
    u'Ф': u'F', u'ф': u'f',
    u'Х': u'H', u'х': u'h',
    u'Ц': u'C', u'ц': u'c',
    u'Ч': u'Č', u'ч': u'č',
    u'Џ': u'Dž', u'џ': u'dž',
    u'Ш': u'Š', u'ш': u'š',
}


# Copied all below from Open-NLLB-stopes: https://github.com/gordicaleksa/Open-NLLB-stopes
UNICODE_PUNCT = {
    "，": ",",
    "。": ".",
    "、": ",",
    "„": '"',
    "”": '"',
    "“": '"',
    "«": '"',
    "»": '"',
    "１": '"',
    "」": '"',
    "「": '"',
    "《": '"',
    "》": '"',
    "´": "'",
    "∶": ":",
    "：": ":",
    "？": "?",
    "！": "!",
    "（": "(",
    "）": ")",
    "；": ";",
    "–": "-",
    "—": " - ",
    "．": ". ",
    "～": "~",
    "’": "'",
    "…": "...",
    "━": "-",
    "〈": "<",
    "〉": ">",
    "【": "[",
    "】": "]",
    "％": "%",
    "►": "-",
}


NON_PRINTING_CHARS_RE = re.compile(
    f"[{''.join(map(chr, list(range(0,32)) + list(range(127,160))))}]"
)


UNICODE_PUNCT_RE = re.compile(f"[{''.join(UNICODE_PUNCT.keys())}]")


PUNCT_OR_NON_PRINTING_CHARS_RE = re.compile(
    (UNICODE_PUNCT_RE.pattern + NON_PRINTING_CHARS_RE.pattern).replace("][", "")
)


def process_line(line):
    line = PUNCT_OR_NON_PRINTING_CHARS_RE.sub("", line)
    line = re.sub(r'\s', '', line)  # Remove all white spaces.
    line = line.replace('-', '')  # Remove all dashes.
    line = line.replace('_', '')  # Remove all underscores.
    line = line.translate(str.maketrans('', '', string.punctuation))  # Remove all punctuation.
    line = line.translate(str.maketrans('', '', '0123456789'))  # Remove all digits.
    return line


def cyrl_vs_latn_worker_func(docs, add_tqdm = False):
    cyrl_cnt_partial = 0
    latn_cnt_partial = 0
    if add_tqdm:
        docs = tqdm(docs)
    for doc in docs:
        doc = process_line(doc)
        cyrl_cnt = sum(list(map(lambda c: c in SR_CYR_TO_LAT_DICT.keys(), doc)))
        cyrl_cnt_partial += cyrl_cnt
        latn_cnt_partial += len(doc) - cyrl_cnt

    return cyrl_cnt_partial, latn_cnt_partial


def count_cyrillic_and_latin_chars(docs, num_workers = 0):
    print(f"Total number of docs: {len(docs)}")

    cyrl_cnt_total = 0
    latn_cnt_total = 0

    if num_workers > 0:
        num_chunks = len(docs)
        chunk_size = num_chunks // num_workers
        docs_chunks = [docs[i:i + chunk_size] for i in range(0, num_chunks, chunk_size)]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(cyrl_vs_latn_worker_func, docs_chunk) for docs_chunk in docs_chunks
            ]
            with tqdm(total=num_chunks) as pbar:
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)
            for future in futures:
                cyrl_cnt_partial, latn_cnt_partial = future.result()
                cyrl_cnt_total += cyrl_cnt_partial
                latn_cnt_total += latn_cnt_partial
    else:
        cyrl_cnt_total, latn_cnt_total = cyrl_vs_latn_worker_func(docs, add_tqdm=True)

    total = cyrl_cnt_total + latn_cnt_total
    print(f"Total number of chars: {total}")
    latn_percentage = latn_cnt_total / total * 100
    cyrl_percentage = cyrl_cnt_total / total * 100
    print(f"Percentage of Latin chars: {latn_percentage:.2f}%")
    print(f"Percentage of Cyrillic chars: {cyrl_percentage:.2f}%")


def visualize_macocu_stats():
    paths = [
        "/hdd/BalkanGPT/MaCoCu/MaCoCu-cnr-1.0.xml",
        "/hdd/BalkanGPT/MaCoCu/MaCoCu-bs-1.0.xml",
        "/hdd/BalkanGPT/MaCoCu/MaCoCu-hr-2.0.xml",
        "/hdd/BalkanGPT/MaCoCu/MaCoCu-sr-1.0.xml"
    ]
    doc_meta_of_interest = ["crawl_date", "lang_distr", "file_type", "lm_score"]
    doc_paragraph_meta_of_interest = ["quality", "lm_score"]
    for path in paths:
        doc_data = defaultdict(list)
        lang_distr = defaultdict(int)
        file_types = defaultdict(int)
        qualities = defaultdict(int)
        langs = defaultdict(int)
        crawl_dates = defaultdict(int)
        paragraphs_data = defaultdict(list)
        print(f"Processing {path}")
        dset = dataset(path)

        debug_file = open(f"debug_langs_{os.path.basename(path).split('-')[1]}.txt", "a")

        length = 0
        for doc in tqdm(dset):
            length += 1

        dset = dataset(path)
        for doc in tqdm(dset, total=length):
            meta = doc.meta
            for key in doc_meta_of_interest:
                if key == "file_type":
                    file_types[meta[key]] += 1
                elif key == "crawl_date":
                    crawl_dates[meta[key]] += 1
                elif key == "lang_distr":
                    lang_distr[meta[key]] += 1
                    expr = eval(meta[key])[0][0]
                    if expr not in ["hbs_cyrl", "hbs_latn", "n/a", "hbs_cyr", "hbs_lat", "mk"]:
                        print(f"lang={meta[key]} ------> {str(doc)}", file=debug_file)
                        debug_file.flush()
                else:
                    assert key == "lm_score"
                    doc_data[key].append(float(meta[key]))

            for paragraph in doc:
                meta = paragraph.meta
                for key in doc_paragraph_meta_of_interest:
                    if key == "quality":
                        qualities[meta[key]] += 1
                    else:
                        assert key == "lm_score"
                        paragraphs_data[key].append(float(meta[key]))

        debug_file.close()

        crawl_dates = sorted(crawl_dates.items(), key=lambda x: x[0])
        plt.bar([date for date, _ in crawl_dates], [count for _, count in crawl_dates])
        plt.xticks(rotation=90)
        plt.close()

        print(f'File types: {file_types}')
        print(f'Qualities: {qualities}')

        plt.hist(doc_data["lm_score"], bins=100)
        plt.title(f"document lm_score for {path}")
        plt.xticks(rotation=90)
        plt.show()
        plt.close()

        plt.hist(paragraphs_data["lm_score"], bins=100)
        plt.title(f"paragraph lm_score for {path}")
        plt.xticks(rotation=90)
        plt.show()
        plt.close()


def count_cyrl_vs_latn_macocu():
    paths = [
        "/hdd/BalkanGPT/MaCoCu/MaCoCu-cnr-1.0.xml",
        "/hdd/BalkanGPT/MaCoCu/MaCoCu-bs-1.0.xml",
        "/hdd/BalkanGPT/MaCoCu/MaCoCu-hr-2.0.xml",
        "/hdd/BalkanGPT/MaCoCu/MaCoCu-sr-1.0.xml"
    ]
    for path in paths:
        print(f"Processing {path}")
        dset = dataset(path)

        length = 0
        for doc in dset:
            length += 1

        dset = dataset(path)
        docs = []
        for doc in tqdm(dset, total=length):
            docs.append(str(doc))

        count_cyrillic_and_latin_chars(docs)


class dataset:  # crawl_date, lang_distr, file_type, lm_score (use thresholding to find outliers)
# paragraph: quality (count), lm_score, sensitive (find yes), lang (find a set of all)
    def __init__(self, path, xml=True):
        self.path = path
        self.file = open(path)
        if xml:
            line = self.file.readline()
            if not line.startswith('<corpus'):
                warnings.warn(
                    "Warning: your prevert might not have a XML header. Define the xml second parameter as False in the constructor if this is the case.")

    def __iter__(self):
        return self

    def __next__(self):
        doc_text = []
        for line in self.file:
            doc_text.append(line)
            if line == '</doc>\n':
                return document(doc_text)
        raise StopIteration


class document:

    def __init__(self, lines):
        self.lines = lines
        self.idx = 1
        self.meta = OrderedDict(re_attr_val.findall(self.lines[0]))
        self.metastr = self.lines[0]

    def __iter__(self):
        return self

    def __next__(self):
        p_text = []
        for line in self.lines[self.idx:]:
            p_text.append(line)
            self.idx += 1
            if line == '</p>\n':
                return paragraph(p_text)
        raise StopIteration

    def __str__(self):
        return ''.join([e for e in self.lines if e[0] != "<"])

    def to_prevert(self):
        lines = self.lines
        firstline = "<doc"
        for key, value in self.meta.items():
            firstline += f' {key}="{str(value)}"'
        firstline += ">\n"
        lines[0] = firstline
        return ''.join([e for e in lines])


class paragraph:

    def __init__(self, lines):
        self.lines = lines
        self.meta = OrderedDict(re_attr_val.findall(self.lines[0]))

    def __str__(self):
        return ''.join(self.lines[1:-1])


def process_macocu():
    paths = [
        "/hdd/BalkanGPT/MaCoCu/MaCoCu-cnr-1.0.xml",
        "/hdd/BalkanGPT/MaCoCu/MaCoCu-bs-1.0.xml",
        "/hdd/BalkanGPT/MaCoCu/MaCoCu-hr-2.0.xml",
        "/hdd/BalkanGPT/MaCoCu/MaCoCu-sr-1.0.xml"
    ]
    for in_path in paths:
        base_dir = os.path.dirname(in_path)
        filename = os.path.basename(in_path)
        out_path = os.path.join(base_dir, filename.split('.')[0] + ".txt")
        dset = dataset(in_path)
        length = 0
        for _ in tqdm(dset):
            length += 1

        dset = dataset(in_path)
        with open(out_path, "w", encoding="utf-8") as fout:
            for doc in tqdm(dset, total=length):
                fout.write(str(doc))
                fout.flush()


def process_riznica_dataset():
    root_path = "/hdd/BalkanGPT/Nikola/data/"
    filename = "riznica.txt"

    import os

    cnt = 0
    with open(os.path.join(root_path, filename), "r") as f:
        lines_buffer = []
        text_chunks = []
        for line in f:
            if line.strip() == "":
                cnt += 1
                text_chunks.append("".join(lines_buffer))
                lines_buffer = []
            else:
                lines_buffer.append(line)

        if len(lines_buffer) > 0:
            cnt += 1
            text_chunks.append("".join(lines_buffer))
            lines_buffer = []

    final_text = '\n'.join(text_chunks)

    with open(os.path.join(root_path, filename), "w") as f:
        f.write(final_text)

    print(cnt)


def pdrs_worker(filepath):
    num_lines = 0
    base_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    out_file = os.path.join(base_dir, filename.split('.')[0] + ".txt")
    with open(out_file, "w", encoding="utf-8") as fout:
        with open(filepath, "r", encoding="utf-8") as fin:
            for tokenlist in parse_incr(fin):
                text = tokenlist.metadata['text']
                num_lines += 1
                fout.write(text + "\n")

    return num_lines, out_file


def download_pdrs_data():  # https://www.clarin.si/repository/xmlui/handle/11356/1752
    root = "/hdd/BalkanGPT/Mihailo/PDRS/"
    file_paths = [os.path.join(root, filename) for filename in os.listdir(root) if filename.endswith(".conllu")]

    num_lines = 0
    num_workers = len(file_paths)
    out_filepaths = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(pdrs_worker, filepath) for filepath in file_paths
        ]
        with tqdm(total=num_workers) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
        for future in futures:
            num_lines_partial, out_file = future.result()
            num_lines += num_lines_partial
            out_filepaths.append(out_file)

    # Merge all out_filepaths into "PDRS.txt" out file.
    with open(os.path.join(root, "PDRS.txt"), "w", encoding="utf-8") as fout:
        for out_file in out_filepaths:
            with open(out_file, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)

    print(f'Number of lines: {num_lines}')


def download_jerteh_hf_data():
    # had to recursively chmod for huggingface cache dir? sudo chmod 777 -R huggingface/ from ~/.cache
    dataset = load_dataset('srwac', cache_dir="/hdd/BalkanGPT/Mihailo/SrWAC")
    print('downloaded srwac dataset')
    print(len(dataset))

    # I manually downloaded the json files as they didn't implement the HF interface for downloading.
    # root = "/hdd/BalkanGPT/Mihailo/"
    # filenames = [filename for filename in os.listdir(root) if filename.endswith(".json")]
    # for filename in filenames:
    #     filepath = os.path.join(root, filename)
    #     out_file = os.path.join(root, filename.split('.')[0] + ".txt")
    #     with open(filepath) as jf, open(out_file, "w") as fout:
    #         sentences = load(jf)["sents"]
    #         for sentence in tqdm(sentences):
    #             fout.write(sentence + "\n")


if __name__ == "__main__":
    # count_cyrl_vs_latn_macocu()
    # visualize_macocu_stats()
    process_macocu()
