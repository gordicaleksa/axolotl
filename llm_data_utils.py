from concurrent.futures import ProcessPoolExecutor
import concurrent
import gzip
import shutil
from tqdm import tqdm
import os
import re
import shlex
import subprocess
import typing as tp
from pathlib import Path
from typing import List



def unzip_worker(in_filepath, out_filepath):
    with gzip.open(in_filepath, 'rb') as f_in:
        with open(out_filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def parallel_unzip(root, filename_buffer, num_workers):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(unzip_worker, os.path.join(root, filename), os.path.join(root, filename.split(".")[0]))
            for filename in filename_buffer
        ]
        with tqdm(total=num_workers) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
        for future in futures:
            future.result()


def unzip_func(num_workers=16):
    root_path = "/hdd/BalkanGPT/Nikola/tmp/"
    for root, _, files in os.walk(root_path):
        filename_buffer = []
        for filename in files:
            if filename.endswith(".gz"):
                filename_buffer.append(filename)
                if len(filename_buffer) == num_workers:
                    print(f"Unzipping {len(filename_buffer)} files...")
                    parallel_unzip(root, filename_buffer, num_workers)
                    filename_buffer = []
    if len(filename_buffer) > 0:
        parallel_unzip(root, filename_buffer, min(num_workers, len(filename_buffer)))


def open_file_cmd(filename: tp.Union[Path, str]) -> str:
    if isinstance(filename, Path):
        filename = str(filename)
    filename = shlex.quote(filename)
    cat = "cat"
    if filename.endswith(".xz"):
        cat = "xzcat"
    if filename.endswith(".gz"):
        cat = "zcat"

    return shlex.join((cat, filename))


def bash_pipefail(*pipe_parts: str) -> str:
    """Run a bash pipelines with "-o pipefail".
    This allows to catch zcat failures.
    Note that it will also generate error if you use "head" in your pipeline.
    The arguments are supposed to be valid bash commands.
    """
    pipe = " | "
    return shlex.join(["/bin/bash", "-o", "pipefail", "-c", pipe.join(pipe_parts)])


def count_lines_or_words(filepath: str, count_lines=True) -> int:
    """
    Count the number of lines in a file.
    """
    result = subprocess.run(
        bash_pipefail(
            open_file_cmd(filepath),
            shlex.join(["wc", "-l" if count_lines else "-w"]),
        ),
        capture_output=True,
        shell=True,
    )
    out = result.stdout.decode("utf-8")
    lines_numbers = [int(line) for line in out.split() if line]
    assert len(lines_numbers) == 1
    return lines_numbers[0]


def count_lines_worker(filepath: str) -> int:
    """
    Count the number of lines in a file.
    """
    return count_lines_or_words(filepath)


def count_words_worker(filepath: str) -> int:
    """
    Count the number of words in a file.
    """
    return count_lines_or_words(filepath, count_lines=False)


def count_in_dir(num_workers=16, count_lines=True):
    root = "/hdd/BalkanGPT/Nikola/data/"
    acc = 0
    filename_buffer = []
    for root, _, files in os.walk(root):
        for filename in files:
            filename_buffer.append(filename)
            if len(filename_buffer) == num_workers:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(count_lines_worker if count_lines else count_words_worker, os.path.join(root, filename))
                        for filename in filename_buffer
                    ]
                    with tqdm(total=num_workers) as pbar:
                        for _ in concurrent.futures.as_completed(futures):
                            pbar.update(1)
                    for future in futures:
                        acc += future.result()
                filename_buffer = []

    if len(filename_buffer) > 0:
        num_workers = min(num_workers, len(filename_buffer))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(count_lines_worker if count_lines else count_words_worker, os.path.join(root, filename))
                for filename in filename_buffer
            ]
            with tqdm(total=num_workers) as pbar:
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)
            for future in futures:
                acc += future.result()

    print(f"Total number of {'lines' if count_lines else 'words'}: {acc}")


def document_length_counter_worker(filepath: str) -> List[str]:
    cnt = 0
    doc_lengths = []
    with open(filepath, "r") as f:
        # Find double newlines without reading the whole file in memory
        for line in f:
            if line != "\n":
                word_count = len(re.findall(r'\b\w+\b', line.strip()))
                cnt += word_count
            else:
                # end of document
                # if cnt <= 50
                doc_lengths.append(cnt)
                cnt = 0  # reset counter

        if cnt > 0:
            doc_lengths.append(cnt)
            cnt = 0  # reset counter

    return doc_lengths


def count_doc_lengths_in_dir(root_path, num_workers=16):
    filename_buffer = []
    doc_lengths = []
    for root, _, files in os.walk(root_path):
        for filename in files:
            filename_buffer.append(filename)
            if len(filename_buffer) == num_workers:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(document_length_counter_worker, os.path.join(root, filename))
                        for filename in filename_buffer
                    ]
                    with tqdm(total=num_workers) as pbar:
                        for _ in concurrent.futures.as_completed(futures):
                            pbar.update(1)
                    for future in futures:
                        doc_lengths.extend(future.result())
                filename_buffer = []

    if len(filename_buffer) > 0:
        num_workers = min(num_workers, len(filename_buffer))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(document_length_counter_worker, os.path.join(root, filename))
                for filename in filename_buffer
            ]
            with tqdm(total=num_workers) as pbar:
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)
            for future in futures:
                doc_lengths.extend(future.result())

    return doc_lengths


def remove_short_doc_from_file_worker(in_filepath: str, out_filepath: str) -> [int, int]:
    DOCUMENT_LEN_THRESHOLD_NUM_WORDS = 20
    cnt = 0
    doc_saved_cnt = 0
    doc_removed_cnt = 0

    document_lines = []
    with open(in_filepath, "r") as f_in, open(out_filepath, "w") as f_out:
        # Find double newlines without reading the whole file in memory
        for line in f_in:
            if line != "\n":
                word_count = len(re.findall(r'\b\w+\b', line.strip()))
                cnt += word_count
                document_lines.append(line)
            else:
                # end of document
                if cnt >= DOCUMENT_LEN_THRESHOLD_NUM_WORDS:
                    doc_saved_cnt += 1
                    f_out.write(''.join(document_lines) + "\n")
                else:
                    doc_removed_cnt += 1

                document_lines = []  # reset document lines
                cnt = 0  # reset counter

        if len(document_lines):
            if cnt >= DOCUMENT_LEN_THRESHOLD_NUM_WORDS:
                f_out.write(''.join(document_lines))
                doc_saved_cnt += 1
            else:
                doc_removed_cnt += 1

        print(f"{in_filepath}: Saved {doc_saved_cnt} documents, removed {doc_removed_cnt} documents")

    return doc_saved_cnt, doc_removed_cnt


def remove_short_docs_from_files_in_dir(num_workers=16):
    root_path = "/hdd/BalkanGPT/Nikola/data/raw/"
    out_dir = "/hdd/BalkanGPT/Nikola/data/filtered/"
    doc_saved_cnt_global = 0
    doc_removed_cnt_global = 0
    filename_buffer = []
    for _, _, files in os.walk(root_path):
        for filename in files:
            filename_buffer.append(filename)
            if len(filename_buffer) == num_workers:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(remove_short_doc_from_file_worker, os.path.join(root_path, filename), os.path.join(out_dir, filename))
                        for filename in filename_buffer
                    ]
                    with tqdm(total=num_workers) as pbar:
                        for _ in concurrent.futures.as_completed(futures):
                            pbar.update(1)
                    for future in futures:
                        doc_saved_cnt, doc_removed_cnt = future.result()
                        doc_saved_cnt_global += doc_saved_cnt
                        doc_removed_cnt_global += doc_removed_cnt
                filename_buffer = []

    if len(filename_buffer) > 0:
        num_workers = min(num_workers, len(filename_buffer))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(remove_short_doc_from_file_worker, os.path.join(root_path, filename), os.path.join(out_dir, filename))
                for filename in filename_buffer
            ]
            with tqdm(total=num_workers) as pbar:
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)
            for future in futures:
                doc_saved_cnt, doc_removed_cnt = future.result()
                doc_saved_cnt_global += doc_saved_cnt
                doc_removed_cnt_global += doc_removed_cnt

    print(f"Total number of documents saved: {doc_saved_cnt_global}")
    print(f"Total number of documents removed: {doc_removed_cnt_global}")
    return doc_saved_cnt_global, doc_removed_cnt_global


if __name__ == "__main__":
    doc_saved_cnt_global, doc_removed_cnt_global = remove_short_docs_from_files_in_dir()

    path_raw = "/hdd/BalkanGPT/Nikola/data/raw"
    doc_lengths_raw = count_doc_lengths_in_dir(path_raw)
    with open("raw_doc_lengths.txt", "w") as f:
        for doc_length in tqdm(doc_lengths_raw):
            f.write(f"{doc_length}\n")

    path_filtered = "/hdd/BalkanGPT/Nikola/data/filtered"
    doc_lengths_filtered = count_doc_lengths_in_dir(path_filtered)
    with open("filtered_doc_lengths.txt", "w") as f:
        for doc_length in tqdm(doc_lengths_filtered):
            f.write(f"{doc_length}\n")

    # # from collections import Counter
    # with open("new_doc_lengths.txt", "r") as f:
    #     doc_lengths = [int(line.strip()) for line in f.readlines() if int(line.strip())]

    print(f'Total number of documents: {len(doc_lengths_raw)}, tokens = {sum(doc_lengths_raw)}')
    print(f'Total number of documents: {len(doc_lengths_filtered)}, tokens = {sum(doc_lengths_filtered)}')

    doc_lengths_filtered_posthoc = [doc_length for doc_length in doc_lengths_raw if doc_length >= 20]

    print(f'Total number of documents (>= 20 words): {len(doc_lengths_filtered_posthoc)}, tokens = {sum(doc_lengths_filtered_posthoc)}')

    # with open("20w_filtered_doc_lengths.txt", "r") as f:
    #     doc_lengths_20w = [int(line.strip()) for line in f.readlines() if int(line.strip())]

    assert len(doc_lengths_filtered) == len(doc_lengths_filtered_posthoc) == doc_saved_cnt_global, f'{len(doc_lengths_filtered)} != {len(doc_lengths_filtered_posthoc)} != {doc_saved_cnt_global}'
    assert len(doc_lengths_raw) == doc_saved_cnt_global + doc_removed_cnt_global, f'{len(doc_lengths_raw)} != {doc_saved_cnt_global} + {doc_removed_cnt_global}'

    # c = Counter(doc_lengths)
    # total = sum(c.values())
    # print(f"Total number of documents: {total}")

    # # Convert number of documents to number of tokens by multiplying by the document length with the number of documents
    # c = Counter({k: k * v for k, v in c.items()})

    # # lowest_2500_keys = sorted(c.keys())[:2500]

    # # # Create a new counter with only those keys
    # # c = Counter({k: c[k] for k in lowest_2500_keys})
    # # # get sum of values
    # # total = sum(c.values())
    # # print(f"Total number of documents (filtered): {total}")

    # # Plot histogram
    # import matplotlib.pyplot as plt
    # import numpy as np

    # plt.bar(c.keys(), c.values())
    # plt.title("Document lengths")
    # plt.xlabel("Document length")
    # plt.ylabel("Frequency")
    # plt.show()
    # print(f"Mean document length: {np.mean(doc_lengths)}")
