import os
import pandas as pd
from backend.load_config import GLOBAL_CONFIG
from backend.utils.spectrum_process import load_spectrum_file

MAX_SPECTRUM_NUM = GLOBAL_CONFIG["identification"]["max_spectrum"]
MAX_FILES = GLOBAL_CONFIG["identification"]["max_files"]
MAX_FILE_SIZE = eval(GLOBAL_CONFIG["identification"]["max_file_size"])

def load_files(
    file_list,
    *,
    is_super_user: bool = False,
    max_spectrum_num: int = MAX_SPECTRUM_NUM,
    max_file_size: int = MAX_FILE_SIZE,
):
    if not file_list:
        raise ValueError("No files provided")

    total_size = sum(os.path.getsize(p) for p in file_list if os.path.exists(p))
    if (not is_super_user) and total_size > max_file_size:
        raise ValueError("File size too large")

    if (not is_super_user) and len(file_list) > MAX_FILES:
        raise ValueError(f"Too many files (max {MAX_FILES})")

    target_zip_file_name = "result.zip"
    if len(file_list) == 1:
        base = os.path.basename(file_list[0])
        target_zip_file_name = f"{base}.zip"

    spectrum_list = []
    for file_name in file_list:
        try:
            loaded = load_spectrum_file(file_name)
        except Exception:
            raise ValueError("Please upload standard file")
        spectrum_list.extend(loaded)
        if (not is_super_user) and len(spectrum_list) > max_spectrum_num:
            raise ValueError(f"Only a maximum of {max_spectrum_num} spectra are allowed to be uploaded")

    titles = [
        s.metadata["compound_name"] if "compound_name" in (s.metadata or {})
        else f"spectrum {i}"
        for i, s in enumerate(spectrum_list)
    ]
    spectrums_df = pd.DataFrame({"title": titles, "spectrum": spectrum_list})
    name_df = spectrums_df[["title"]]
    return spectrums_df, name_df, target_zip_file_name
