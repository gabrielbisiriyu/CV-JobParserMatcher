import openai
from pydantic import BaseModel
import json
from pypdf import PdfReader
import docx2txt
import os
from decouple import config
import asyncio
from typing import List


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
    companyInfo: List[CompanyDetails]
    jobTitle: str
    requiredSkills: List[str]
    roles_or_responsibilities: List[str]


class JobPostingExtractor:
    def __init__(self):
        api_key = config("OPENAI_API_KEY")
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def extract_job_info(self, text: str):
        # Create the JSON schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "companyInfo": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "companyName": {"type": "string"},
                            "location": {"type": "string"},
                            "website": {"type": "string"},
                            "telephoneNumber": {"type": "string"}
                        },
                        "required": ["companyName", "location", "website", "telephoneNumber"],
                        "additionalProperties": False
                    }
                },
                "jobTitle": {"type": "string"},
                "requiredSkills": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "roles_or_responsibilities": {
                    "type": "array",
                    "items": { "type": "string",
                "description": "One individual skill or technology only, not a sentence"}
                }
            },
            "required": ["companyInfo", "jobTitle", "requiredSkills", "roles_or_responsibilities"],
            "additionalProperties": False
        }

        prompt = """Parse this Job Posting to find relevant information about the job.

                Extract the following information and return it in the specified JSON format:
                - Company information (name, location, website, phone number)
                - Job title
                -  Required skills: ONLY atomic, individual skills or technologies.
                     Example: ["JavaScript", "React", "MongoDB", "CI/CD"].
                     Do NOT include years of experience, full sentences, or descriptions.
                - Roles and responsibilities (job duties, what the person will do)

                Important guidelines:
                - Use empty strings ("") for missing information, never use null or omit fields
                - For arrays, use empty arrays ([]) if no information is found
                - Company info should be a single-item array with the company's details
                - Be thorough in extracting all mentioned skills and requirements
                - Extract detailed responsibilities and job duties
                - Include both technical and soft skills in requiredSkills
                - Separate individual responsibilities into array items

                Job posting text:"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4o" for better quality
                messages=[
                    {"role": "system", "content": "You are a job posting parsing assistant that extracts structured information from job descriptions."},
                    {"role": "user", "content": f"{prompt}\n\n{text}"}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "job_info",
                        "schema": schema,
                        "strict": True
                    }
                },
                temperature=0.1,
                max_tokens=4000
            )
            
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            raise RuntimeError(f"Failed to extract job info: {e}")


