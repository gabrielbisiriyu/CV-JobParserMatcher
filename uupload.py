from huggingface_hub import HfApi, HfFolder, Repository

from huggingface_hub import upload_folder

upload_folder(
    folder_path="cv_jd_finetuned_model2",
    repo_id="ennygaebs/cv-job-matcher",
    repo_type="model",
    commit_message="Initial commit of fine-tuned SBERT"
)
