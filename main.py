from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
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
from sqlalchemy import select, update, func
from uuid import UUID
import torch

#  GLOBAL matcher variable, not instantiated yet
matcher = None

app = FastAPI()

#  INIT matcher ONCE when app starts
@app.on_event("startup")
def load_matcher():
    global matcher
    matcher = CVJobMatcher()  # Loaded only once at startup
    print("🧠 CVJobMatcher loaded globally on startup")



@app.post("/parse_cv/")
async def parse_cv(user_id: str = Form(...), file: UploadFile = File(...)):
    global matcher  # ⬅️ Use the global instance
    suffix = Path(file.filename).suffix
    if suffix not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only .pdf and .docx allowed.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        extractor = ResumeExtractor()
        text_cv = extract_text_cv(tmp_path)
        text_cv_hash = hashlib.sha256(text_cv.encode("utf-8")).hexdigest()
        data_cv = extractor.extract_all(text_cv)

        # Process experience
        contextual_experience = []
        for exp in data_cv["experience"]:
            if isinstance(exp, dict):
                exp_data = f"{exp.get('job_title', 'Unknown')} at {exp.get('company', 'Unknown')}: {exp.get('responsibilities', '')}"
                contextual_experience.append(exp_data)

        exp_text = " ".join(contextual_experience)
        skill_list = data_cv["skills"]
        cv_emb, skill_emb, exp_emb = matcher.cv_DOCnFIELD_level_embeddings(text_cv, skill_list, exp_text)

        async with SessionLocal() as session:
            result = await session.execute(select(ParsedCV).where(ParsedCV.user_id == user_id))
            existing_cv = result.scalar_one_or_none()

            if existing_cv:
                await session.execute(
                    update(ParsedCV)
                    .where(ParsedCV.user_id == user_id)
                    .values(
                        text_hash=text_cv_hash,
                        parsed_fields=data_cv,
                        cv_emb=cv_emb.tolist(),
                        skill_emb=skill_emb.tolist(),
                        exp_emb=exp_emb.tolist(),
                    )
                )
            else:
                cv_record = ParsedCV(
                    user_id=user_id,
                    text_hash=text_cv_hash,
                    parsed_fields=data_cv,
                    cv_emb=cv_emb.tolist(),
                    skill_emb=skill_emb.tolist(),
                    exp_emb=exp_emb.tolist(),
                )
                session.add(cv_record)

            await session.commit()

        return {
            "hash": text_cv_hash,
            "parsed_cv": data_cv,
            "embeddings": {
                "cv_emb": cv_emb.tolist(),
                "skill_emb": skill_emb.tolist(),
                "exp_emb": exp_emb.tolist()
            }
        }

    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error parsing CV.")
    finally:
        os.remove(tmp_path)


@app.post("/parse_job/")
async def parse_job(company_id: str = Form(...), file: UploadFile = File(...)):
    global matcher  # ⬅️ Use the global instance
    suffix = Path(file.filename).suffix
    if suffix not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only .pdf, .docx, .txt allowed.")

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
        jd_emb, req_skill_emb, role_emb = matcher.job_DOCnFIELD_level_embeddings(text_job, r_skills, res_text)

        async with SessionLocal() as session:
            count_query = await session.execute(
                func.count().select().where(ParsedJob.company_id == company_id)
            )
            job_count = count_query.scalar()

            if job_count >= 2:
                raise HTTPException(status_code=403, detail="Job limit reached (10). Delete old ones to add more.")

            job_record = ParsedJob(
                company_id=company_id,
                text_hash=text_job_hash,
                job_text=text_job,
                parsed_fields=data_job,
                cv_emb=jd_emb.tolist(),
                skill_emb=req_skill_emb.tolist(),
                exp_emb=role_emb.tolist()
            )
            session.add(job_record)
            await session.commit()
            job_id = str(job_record.id)

        return {
            "job_id": job_id,
            "hash": text_job_hash,
            "parsed_job": data_job,
            "job_text": text_job,
            "embeddings": {
                "cv_emb": jd_emb.tolist(),
                "skill_emb": req_skill_emb.tolist(),
                "exp_emb": role_emb.tolist()
            }
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error parsing job post.")
    finally:
        os.remove(tmp_path)


@app.post("/match_cv_to_jobs/")
async def match_cv_to_jobs(cv_hash: str, top_n: int = Query(5, ge=1, le=50)):
    global matcher  # ⬅️ Use the global instance
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
                "job_title": job.parsed_fields.get("jobTitle", {}),
                **score
            })

        results.sort(key=lambda x: -x["combined_score"])
        return results[:top_n]


@app.post("/match_job_to_cvs/")
async def match_job_to_cvs(job_hash: str, top_n: int = Query(5, ge=1, le=50)):
    global matcher  # ⬅️ Use the global instance
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


@app.delete("/delete_job/{job_id}")
async def delete_job(job_id: UUID):
    async with SessionLocal() as session:
        result = await session.execute(select(ParsedJob).where(ParsedJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        await session.delete(job)
        await session.commit()
        return {"detail": f"Job {job_id} deleted successfully"}


@app.get("/jobs/company/{company_id}")
async def get_company_jobs(company_id: str, limit: int = Query(5), offset: int = Query(0)):
    async with SessionLocal() as session:
        try:
            result = await session.execute(
                select(ParsedJob)
                .where(ParsedJob.company_id == company_id)
                .offset(offset)
                .limit(limit)
            )
            jobs = result.scalars().all()

            return {
                "count": len(jobs),
                "offset": offset,
                "limit": limit,
                "jobs": [
                    {
                        "id": str(job.id),
                        "text_hash": job.text_hash,
                        "parsed_fields": job.parsed_fields,
                        "created_at": job.created_at,
                    }
                    for job in jobs
                ]
            }

        except Exception as e:
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Error retrieving job posts.")
