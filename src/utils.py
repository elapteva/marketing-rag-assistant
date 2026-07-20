from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".pptx"}
MAX_FILE_SIZE_MB = 25

def validate_uploaded_file(filename: str, size_bytes: int) -> tuple[bool, str]:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type: {extension}"
    if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File exceeds {MAX_FILE_SIZE_MB} MB."
    return True, "File is valid."

