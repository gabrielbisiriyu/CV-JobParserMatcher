
from fastapi import FastAPI, UploadFile, File, HTTPException,Query
import tempfile
from extractor_resume2 import ResumeExtractor, extract_text_cv
from extractor_job2 import JobPostingExtractor, extract_text_job
import os 
from pathlib import Path
import logging
import hashlib
import json
import traceback
from matcher3 import CVJobMatcher
from db import SessionLocal
from models import ParsedCV, ParsedJob
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select
import torch



# Load OpenAI key from environment variable (or fallback)
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "mock-key")

matcher = CVJobMatcher(model_name='ennygaebs/cv-job-matcher')


app = FastAPI()

@app.post("/parse_cv/")
async def parse_cv(file: UploadFile = File(...)):
    # Save uploaded file to a temporary file
    suffix = Path(file.filename).suffix
    if suffix not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only .pdf and .docx allowed.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        extractor = ResumeExtractor()
        text_cv = extract_text_cv(tmp_path)
        text_cv_hash = hashlib.sha256(text_cv.encode('utf-8')).hexdigest()
        data_cv =extractor.extract_all(text_cv)

        
        contextual_experience = []
        for exp in data_cv["experience"]:
            if isinstance(exp, dict):
                exp_data = f"{exp.get('job_title', 'Unknown')} at {exp.get('company', 'Unknown')}: {exp.get('responsibilities', '')}"
                contextual_experience.append(exp_data)
            else:
                print(f"⚠️ Skipping malformed experience entry: {exp}")

        exp_text = ' '.join(contextual_experience)
        skill_list = data_cv["skills"]
        cv_emb,skill_emb,exp_emb = matcher.cv_DOCnFIELD_level_embeddings(text_cv,skill_list,exp_text)
        #embeddings_cv = {"cv_emb":cv_emb, "skill_emb":skill_emb,"exp_emb":exp_emb} 
             
        #  TODO: store to database (we'll do that in next step)
        async with SessionLocal() as session:
            try:
                cv_record = ParsedCV(
                    text_hash=text_cv_hash,
                    parsed_fields=data_cv,
                    cv_emb=cv_emb.tolist(),
                    skill_emb=skill_emb.tolist(),
                    exp_emb=exp_emb.tolist()
                )
                session.add(cv_record)
                await session.commit()
            except IntegrityError:
                await session.rollback()
                logging.warning("Duplicate CV detected. Skipping insert.")

        return {
            "hash": text_cv_hash,
            "parsed_cv": data_cv,
            "embeddings": {
                "cv_emb": cv_emb.tolist(),
                "skill_emb": skill_emb.tolist(),
                "exp_emb": exp_emb.tolist()
            }
        }

        #return {"hash": text_hash, "parsed_cv": data}
        #return {"hash": text_cv_hash, "parsed_cv": data_cv, "embeddings":embeddings_cv}
    
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error parsing CV.")
    
    finally:
        os.remove(tmp_path)

@app.post("/parse_job/")
async def parse_job(file: UploadFile = File(...)):
    # Save uploaded file to a temporary file
    suffix = Path(file.filename).suffix
    if suffix not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only .pdf and .docx allowed.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        extractor = JobPostingExtractor()
        text_job = extract_text_job(tmp_path)
        text_job_hash = hashlib.sha256(text_job.encode('utf-8')).hexdigest()
        data_job = extractor.extract_job_info(text_job)
        r_skills = data_job["requiredSkills"]
        res_text = ' '.join(data_job["roles_or_responsibilities"])
        jd_emb,req_skill_emb,role_emb = matcher.job_DOCnFIELD_level_embeddings(text_job,r_skills,res_text)
        #embeddings_jd = {"cv_emb":jd_emb, "skill_emb":req_skill_emb,"exp_emb":role_emb} 

        # 🟩 TODO: store to database (next)

        async with SessionLocal() as session:
            try:
                job_record = ParsedJob(
                    text_hash=text_job_hash,
                    parsed_fields=data_job,
                    cv_emb=jd_emb.tolist(),
                    skill_emb=req_skill_emb.tolist(),
                    exp_emb=role_emb.tolist()
                )
                session.add(job_record)
                await session.commit()
            except IntegrityError:
                await session.rollback()
                logging.warning("Duplicate Job detected. Skipping insert.")

        return {
            "hash": text_job_hash,
            "parsed_job": data_job,
            "embeddings": {
                "cv_emb": jd_emb.tolist(),
                "skill_emb": req_skill_emb.tolist(),
                "exp_emb": role_emb.tolist()
            }
        }

        #return {"hash": text_job_hash, "parsed_job": data_job, "embeddings":embeddings_jd}
    
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error parsing job post.")

    finally:
        os.remove(tmp_path)   



@app.post("/match_cv_to_jobs/")
async def match_cv_to_jobs(cv_hash: str, top_n: int = Query(5, ge=1, le=50)):
    async with SessionLocal() as session:
        cv_row = await session.execute(select(ParsedCV).where(ParsedCV.text_hash == cv_hash))
        cv = cv_row.scalar_one_or_none()
        if not cv:
            raise HTTPException(status_code=404, detail="CV not found")

        jobs_result = await session.execute(select(ParsedJob))
        jobs = jobs_result.scalars().all()

        results = []
        cv_emb = torch.tensor(cv.cv_emb)
        skill_emb = torch.tensor(cv.skill_emb)
        exp_emb = torch.tensor(cv.exp_emb)

        for job in jobs:
            jd_emb = torch.tensor(job.cv_emb)
            r_skill_emb = torch.tensor(job.skill_emb)
            role_emb = torch.tensor(job.exp_emb)

            score = matcher.match(cv_emb, jd_emb, skill_emb, r_skill_emb, exp_emb, role_emb)
            results.append({
                "job_id": job.id,
                "text_hash": job.text_hash,
                "created_at": job.created_at,
                "job_title": job.parsed_fields.get("companyInfo", {}),
                **score
            })

        results.sort(key=lambda x: -x["combined_score"])
        return results[:top_n]

@app.post("/match_job_to_cvs/")
async def match_job_to_cvs(job_hash: str, top_n: int = Query(5, ge=1, le=50)):
    async with SessionLocal() as session:
        job_row = await session.execute(select(ParsedJob).where(ParsedJob.text_hash == job_hash))
        job = job_row.scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        cvs_result = await session.execute(select(ParsedCV))
        cvs = cvs_result.scalars().all()

        results = []
        jd_emb = torch.tensor(job.cv_emb)
        r_skill_emb = torch.tensor(job.skill_emb)
        role_emb = torch.tensor(job.exp_emb)

        for cv in cvs:
            cv_emb = torch.tensor(cv.cv_emb)
            skill_emb = torch.tensor(cv.skill_emb)
            exp_emb = torch.tensor(cv.exp_emb)

            score = matcher.match(cv_emb, jd_emb, skill_emb, r_skill_emb, exp_emb, role_emb)
            results.append({
                "cv_id": cv.id,
                "candidate": cv.parsed_fields.get("personalInfo", {}),
                **score
            })

        results.sort(key=lambda x: -x["combined_score"])
        return results[:top_n]