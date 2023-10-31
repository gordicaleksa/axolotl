import os
from PyPDF2 import PdfReader, PdfWriter

import numpy as np
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf


if __name__ == "__main__":
    root_path = "/home/aleksa/Downloads/"
    filename = "Disertacija_13653.pdf"
    filepath = os.path.join(root_path, filename)
    new_filename = f"test.pdf"
    new_filepath = os.path.join(root_path, new_filename)

    # reader = PdfReader(filepath)
    # writer = PdfWriter()
    # pages_of_interest = list(np.arange(8, 14, dtype=int))
    # for page in pages_of_interest:
    #     writer.add_page(reader.pages[int(page)])
    # with open(new_filepath, "wb") as out:
    #     writer.write(out)

    filepath = new_filepath

    # elements = partition(filepath)
    # print("\n\n".join([str(el) for el in elements][:10]))

    elements = partition_pdf(
        filename=filepath, infer_table_structure=True, ocr_languages="eng+srp", extract_images_in_pdf=True, strategy="hi_res", image_output_dir_path=".")
    tables = [el for el in elements if el.category == "Table"]

    print(tables[0].text)
    print(tables[0].metadata.text_as_html)