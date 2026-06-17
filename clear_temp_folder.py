import os

def clear_directory_files(directory_path):
    """
    Deletes all files within the specified directory.
    Subdirectories and their contents are not affected.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            except FileNotFoundError:
                print(f"File not found (probably deleted by another process): {file_path}")
            except OSError as e:
                print(f"Error removing file {file_path}: {e}")

def temp_clr():
    """
    Clears the temp folder of the repository to prevent accumulation in memo.
    """
    if os.path.exists("tmp"):
        clear_directory_files("tmp")