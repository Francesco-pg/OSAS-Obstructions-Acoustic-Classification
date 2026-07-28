# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:09:26 2025
@author: Francesco
"""

import os
import shutil 
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# def scan_files_flat(target_path: str, 
#                     extensions: list[str], 
#                     outpath: str,
#                     gen_id: bool = True,
#                     id_prefix: str = "OSA", 
#                     id_digits: int = 4,
#                     classes: dict = None) -> None:
#     """
#     Scans a SINGLE folder for files with specified extensions and saves details in an Excel file.
#     """
#     target_path = Path(target_path)
#     file_list = []
    
#     files_to_process = [file for file in os.listdir(target_path) 
#                         if (target_path / file).is_file() and 
#                         any(file.lower().endswith(ext.lower()) for ext in extensions)]

#     for file in tqdm(files_to_process, desc="Scanning files"):
#         if not (target_path / file).is_file():
#             continue
        
#         if not any(file.lower().endswith(ext.lower()) for ext in extensions):
#             continue

#         file_path = target_path / file
#         file_name = file                           
#         file_folder = target_path.name             

#         if classes is not None:
#             name_clean = file_name.replace(" ", "").lower()
#             lab = None
#             for key, value in classes.items():
#                 if key.lower() in name_clean:
#                     lab = value
#                     break
#         else:
#             lab = None

#         file_list.append((file_path, file_folder, file_name, lab))

#     df = pd.DataFrame(file_list, columns=["file_path", "file_folder", "file_name", "category"])

#     if gen_id:
#         df["id"] = [f"{id_prefix}{str(i+1).zfill(id_digits)}" for i in range(len(df))]

#     df.to_excel(outpath, index=False)


def scan_files(target_path: str, 
               extensions: list[str], 
               outpath: str,
               gen_id: bool = True,
               id_prefix: str = "OSA", 
               id_digits: int = 4,
               classes: dict = None) -> None:
    """
    Scans a directory for files with specified extensions and saves details in an Excel file.
    """
    file_list = []
    
    files_to_process = []
    for root, par, files in os.walk(target_path):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in extensions):
                files_to_process.append((root, file))

    for root, file in tqdm(files_to_process, desc="Scanning files"):
        file_path = Path(root) / file
        file_folder = "\\".join(file_path.parts[-3:-1])  
        file_name = file                                 
        c = file_path.parts[-3].replace(" ", "").lower()
        
        lab = classes.get(c, "unknown") if classes else "unknown"
        
        file_list.append((file_path, file_folder, file_name, lab))

    df = pd.DataFrame(file_list, columns=["file_path", "file_folder", "file_name", "category"])

    if gen_id:
        df["id"] = [f"{id_prefix}{str(i+1).zfill(id_digits)}" for i in range(len(df))]  

    df.to_excel(outpath, index=False)

    
# def count_files(target_path: str, formats: list[str]) -> int:
#     """
#     Counts files with specific formats in a directory.
#     """
#     total_files = 0
#     format_counts: dict[str, int] = {}

#     files_to_count = []
#     for _, _, files in os.walk(target_path):
#         for file in files:
#             files_to_count.append(file)
            
#     for file in tqdm(files_to_count, desc="Counting files"):
#         file_extension = file.split('.')[-1].lower()
#         if formats == ["any"] or file_extension in formats:
#             total_files += 1
#             format_counts[file_extension] = format_counts.get(file_extension, 0) + 1

#     if formats == ["any"]:
#         print(f"Total number of files: {total_files}")
#     else:
#         print(f"Total number of files with specified formats: {total_files}")
    
#     for ext, count in format_counts.items():
#         print(f"{count} .{ext} files")

#     return total_files


# def copy_and_rename_files(excel_path: str, target_path: str, new_root_path: str) -> None:
#     """
#     Copies and renames files based on an Excel file mapping, with a confirmation prompt.
#     """
#     print(f"\n⚠️ WARNING: You are about to copy and rename files.\n")
#     print(f"📂 Source directory: {target_path}")
#     print(f"📊 Excel file: {excel_path}")
#     print(f"💾 Destination directory: {new_root_path}\n")
    
#     confirmation = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
#     if confirmation != "yes":
#         print("Operation cancelled.")
#         return

#     df = pd.read_excel(excel_path)

#     for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Copying and renaming files"):
#         original_file_path = row["file_path"]
#         original_file_name = row["file_name"]
#         file_id = row["id"]
#         original_folder = row["file_folder"]
#         file_extension = Path(original_file_name).suffix

#         new_file_name = f"{file_id}{file_extension}"
#         new_folder_path = os.path.join(new_root_path, original_folder)
#         os.makedirs(new_folder_path, exist_ok=True)

#         new_file_path = os.path.join(new_folder_path, new_file_name)

#         if os.path.exists(original_file_path):
#             shutil.copy2(original_file_path, new_file_path)
#         else:
#             print(f"❌ File not found: {original_file_path}")


def concatenate_dataframes(dataframes, id_columns):
    """
    Concatenate multiple dataframes based on shared ID columns.
    """
    if not dataframes:
        raise ValueError("The dataframes list is empty. Provide at least one dataframe.")
    
    result_df = dataframes[0]
    
    for df in dataframes[1:]:
        result_df = pd.merge(result_df, df, on=id_columns, how='left')
    
    return result_df


def find_file_paths(root_folder, target_filename):
    """
    Finds all full paths of a specific filename within a root folder.
    """
    matches = []
    for root, dirs, files in os.walk(root_folder):
        if target_filename in files:
            full_path = os.path.join(root, target_filename)
            matches.append(full_path)
    return matches


def extract_class_and_subject(file_name: str) -> dict:
    """
    Estrae l'ID del soggetto e la classe dal nome del file.
    """
    parts = file_name.lower().split(" ")
    
    if len(parts) < 2:
        parts = file_name.lower().split("_")
        
    if len(parts) < 2:
        return {"subject_id": parts[0].replace(".wav", ""), "class": "unknown"}

    subject_id = parts[0]
    raw_class = parts[1]

    if "palato+secretivo" in raw_class:
        class_type = "secretivo"
    elif raw_class in ["tonsilla", "tonsille"]:
        class_type = "tonsilla"
    else:
        class_type = raw_class

    return {"subject_id": subject_id, "class": class_type}


def scan_files_for_classes(target_path: str, extensions: list[str], outpath: str) -> None:
    """
    Scansiona una directory per file con estensioni specificate e crea un dataframe con 
    soggetto, classe e percorso del file. Salva i risultati in un file Excel.
    """
    file_list = []
    
    target_path = Path(target_path)
    files_to_process = []
    for ext in extensions:
        files_to_process.extend(list(target_path.rglob(f"*{ext}")))
    
    for file in tqdm(files_to_process, desc="Scanning files for classes"):
        file_path = file
        file_name = file.name
        file_id = file.stem
        
        file_info = extract_class_and_subject(file_name)
        
        file_list.append((file_path, file_id, file_info["subject_id"], file_info["class"]))
    
    df = pd.DataFrame(file_list, columns=["file_path", "id", "subject_id", "class"])
    df['class'] = df['class'].replace({"tonsilla": "tonsille", "palato+secretivo": "secretivo"})
    
    df.to_excel(outpath, index=False)