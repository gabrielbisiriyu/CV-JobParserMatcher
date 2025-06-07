from google import genai
from pydantic import BaseModel
import json
from pypdf import PdfReader
import docx2txt
import os
from decouple import config



def extract_text_job(file_path: str) -> str:
    try:
        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
            return text

        elif file_path.endswith(".docx"):
            text = docx2txt.process(file_path)
            return text

        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            return text

        else:
            raise ValueError("Unsupported file type. Only .pdf, .docx, and .txt are supported.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {file_path}: {e}")


class CompanyDetails(BaseModel):
    companyName: str
    location: str
    website: str 
    telephoneNumber: str


class JobInfo(BaseModel):
    companyInfo: list[CompanyDetails]
    requiredSkills: list[str]
    roles_or_responsibilities: list[str]
    


class JobPostingExtractor:
    def __init__(self):
        api_key = config("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def extract_job_info(self, text):
        prompt = "Parse this Job Posting to find relevant information about the job.."

        try:
            response = self.client.models.generate_content(
                model='gemini-1.5-flash',
                contents=[prompt, text],
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': JobInfo
                }
            )
            return json.loads(response.text)

        except Exception as e:
            raise RuntimeError(f"Failed to extract job info: {e}")
