import os
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

# ==========================
# CLASS: ExcelUploader
# ==========================
class ExcelUploader:
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    def validate_file(self, filename: str):
        """Ensure uploaded file is an Excel file"""
        if not filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload an Excel file (.xlsx or .xls)"
            )

    def generate_filename(self, original_name: str) -> str:
        """Generate a unique filename with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{original_name}"

    async def save_file(self, file: UploadFile) -> str:
        """Save the uploaded file to disk and return file path"""
        self.validate_file(file.filename)
        # saved_filename = self.generate_filename(file.filename)
        file_path = os.path.join(self.upload_dir, "niger_delta.xlsx")

        try:
            with open(file_path, "wb") as f:
                f.write(await file.read())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

        return file_path

