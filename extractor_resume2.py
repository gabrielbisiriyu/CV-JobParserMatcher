from google import genai
from pydantic import BaseModel
import json
from pypdf import PdfReader
import docx2txt
import os
from decouple import config
import asyncio


def extract_text_cv(file_path: str) -> str:
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


class PersonalInformation(BaseModel):
    name: str
    location: str
    emailAddress: str
    github: str
    linkedin: str
    telephoneNumber: str


class WorkExperience(BaseModel):
    company: str
    jobTitle: str
    timeWorked: str
    responsibilities: str


class Educatiom(BaseModel):  # typo preserved as in your original
    school: str
    degree: str
    field: str


class ResumeInfo(BaseModel):
    personalInfo: list[PersonalInformation]
    educatipn: list[Educatiom]  # typo preserved
    experience: list[WorkExperience]
    skills: list[str]
    certificates: list[str]


class ResumeExtractor:
    def __init__(self):
        api_key = config("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

    async def extract_all(self, text: str):
        prompt = "Parse this resume to find relevant information about the candidate."

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model='gemini-1.5-flash',
                contents=[prompt, text],
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': ResumeInfo
                }
            )
            return json.loads(response.text)

        except Exception as e:
            raise RuntimeError(f"Failed to extract resume info: {e}")
