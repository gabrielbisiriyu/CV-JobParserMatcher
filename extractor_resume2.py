from google import genai
from pydantic import BaseModel
import json
from pypdf import PdfReader
import docx2txt
import os
from decouple import config
import asyncio
import json
import openai
from pydantic import BaseModel
from typing import List




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


import asyncio
import json
import openai
from pydantic import BaseModel
from decouple import config
from typing import List

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
    personalInfo: List[PersonalInformation]
    educatipn: List[Educatiom]  # typo preserved
    experience: List[WorkExperience]
    skills: List[str]
    certificates: List[str]


class ResumeExtractor:
    def __init__(self):
        api_key = config("OPENAI_API_KEY")
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def extract_all(self, text: str):
        # Create the JSON schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "personalInfo": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "location": {"type": "string"},
                            "emailAddress": {"type": "string"},
                            "github": {"type": "string"},
                            "linkedin": {"type": "string"},
                            "telephoneNumber": {"type": "string"}
                        },
                        "required": ["name", "location", "emailAddress", "github", "linkedin", "telephoneNumber"],
                        "additionalProperties": False
                    }
                },
                "educatipn": {  # typo preserved
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "school": {"type": "string"},
                            "degree": {"type": "string"},
                            "field": {"type": "string"}
                        },
                        "required": ["school", "degree", "field"],
                        "additionalProperties": False
                    }
                },
                "experience": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "company": {"type": "string"},
                            "jobTitle": {"type": "string"},
                            "timeWorked": {"type": "string"},
                            "responsibilities": {"type": "string"}
                        },
                        "required": ["company", "jobTitle", "timeWorked", "responsibilities"],
                        "additionalProperties": False
                    }
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "certificates": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["personalInfo", "educatipn", "experience", "skills", "certificates"],
            "additionalProperties": False
        }

        prompt = """Parse this resume to find relevant information about the candidate.

                Extract the following information and return it in the specified JSON format:
                - Personal information (name, location, email, github, linkedin, phone)
                - Education (school, degree, field of study)
                - Work experience (company, job title, time worked, responsibilities)
                - Skills (list of technical and soft skills)
                - Certificates (list of certifications)

                Important guidelines:
                - Use empty strings ("") for missing information, never use null or omit fields
                - For arrays, use empty arrays ([]) if no information is found
                - Personal info should be a single-item array with the candidate's details
                - Time worked should be in format like "Jan 2020 - Dec 2022" or "2020-2022"
                - Be thorough in extracting all mentioned skills and certifications
                - Extract complete responsibility descriptions for work experience

                Resume text:"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4o" for better quality
                messages=[
                    {"role": "system", "content": "You are a resume parsing assistant that extracts structured information from resumes."},
                    {"role": "user", "content": f"{prompt}\n\n{text}"}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "resume_info",
                        "schema": schema,
                        "strict": True
                    }
                },
                temperature=0.1,
                max_tokens=4000
            )
            
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            raise RuntimeError(f"Failed to extract resume info: {e}")


# Alternative version using function calling (if structured outputs aren't available)
class ResumeExtractorFunctionCalling:
    def __init__(self):
        api_key = config("OPENAI_API_KEY")
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def extract_all(self, text: str):
        functions = [
            {
                "name": "extract_resume_info",
                "description": "Extract structured information from a resume",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "personalInfo": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "location": {"type": "string"},
                                    "emailAddress": {"type": "string"},
                                    "github": {"type": "string"},
                                    "linkedin": {"type": "string"},
                                    "telephoneNumber": {"type": "string"}
                                }
                            }
                        },
                        "educatipn": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "school": {"type": "string"},
                                    "degree": {"type": "string"},
                                    "field": {"type": "string"}
                                }
                            }
                        },
                        "experience": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "company": {"type": "string"},
                                    "jobTitle": {"type": "string"},
                                    "timeWorked": {"type": "string"},
                                    "responsibilities": {"type": "string"}
                                }
                            }
                        },
                        "skills": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "certificates": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["personalInfo", "educatipn", "experience", "skills", "certificates"]
                }
            }
        ]

        prompt = f"""Parse this resume to find relevant information about the candidate.

Extract all available information including:
- Personal details (name, location, email, social profiles, phone)
- Education history (schools, degrees, fields of study)
- Work experience (companies, job titles, duration, responsibilities)
- Skills (technical and soft skills)
- Certifications

Use empty strings for missing information and empty arrays for missing lists.

Resume text:
{text}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a resume parsing assistant."},
                    {"role": "user", "content": prompt}
                ],
                functions=functions,
                function_call={"name": "extract_resume_info"},
                temperature=0.1,
                max_tokens=4000
            )
            
            function_call = response.choices[0].message.function_call
            return json.loads(function_call.arguments)

        except Exception as e:
            raise RuntimeError(f"Failed to extract resume info: {e}")