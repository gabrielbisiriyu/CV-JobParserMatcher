from sentence_transformers import SentenceTransformer, util
import numpy as np
import re

'''
THIS IS A TEST MATCHER FROM matcher.py WITH A LITTLE TWEAK IN THE EXPERIENCE SECTION
 I ALSO TWEAKED THE SCORING

'''




class CVJobMatcher:
    def __init__(self, model_name='ennygaebs/cv-job-matcher'):
        #model_name='cv_jd_finetuned_model2'
        #model_name='ennygaebs/cv-job-matcher'
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model '{model_name}' loaded.")
        
    def _split_into_sentences(self, text):
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences

    def _chunk_sentences(self, sentences, max_words=350):
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            words = sentence.split()
            if current_length + len(words) > max_words:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = words
                current_length = len(words)
            else:
                current_chunk.extend(words)
                current_length += len(words)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def cv_DOCnFIELD_level_embeddings(self, cv_text, skills, experience, max_words=350):
        # Encode each skill separately and average
        skill_embeddings = self.model.encode(skills, convert_to_tensor=True,batch_size=64)
        if len(skill_embeddings.shape) > 1:
            skill_embeddings = skill_embeddings.mean(dim=0)

        # Experience context
        experience_text =  experience
        experience_sentences = self._split_into_sentences(experience_text)
        experience_chunks = self._chunk_sentences(experience_sentences, max_words=max_words)
        experience_embeddings = self.model.encode(experience_chunks, convert_to_tensor=True,batch_size=64)
        if len(experience_embeddings.shape) > 1:
            experience_embeddings = experience_embeddings.mean(dim=0)

        # Chunk CV text and encode
        cv_sentences = self._split_into_sentences(cv_text)
        cv_chunks = self._chunk_sentences(cv_sentences, max_words=max_words)
        cv_embeddings = self.model.encode(cv_chunks, convert_to_tensor=True,batch_size=64)
        if len(cv_embeddings.shape) > 1:
            cv_embeddings = cv_embeddings.mean(dim=0)

        return cv_embeddings, skill_embeddings, experience_embeddings

    def job_DOCnFIELD_level_embeddings(self, job_text, required_skills, responsibilities, max_words=300):
        # Encode each required skill and average
        required_skill_embeddings = self.model.encode(required_skills, convert_to_tensor=True,batch_size=64)
        if len(required_skill_embeddings.shape) > 1:
            required_skill_embeddings = required_skill_embeddings.mean(dim=0)

        # Experience context
        role_text =  responsibilities
        role_sentences = self._split_into_sentences(role_text)
        role_chunks = self._chunk_sentences(role_sentences, max_words=max_words)
        role_embeddings = self.model.encode(role_chunks, convert_to_tensor=True,batch_size=64)
        if len(role_embeddings.shape) > 1:
            role_embeddings = role_embeddings.mean(dim=0)

        # Chunk job text and encode
        jd_sentences = self._split_into_sentences(job_text)
        jd_chunks = self._chunk_sentences(jd_sentences, max_words=max_words)
        jd_embeddings = self.model.encode(jd_chunks, convert_to_tensor=True,batch_size=64)
        if len(jd_embeddings.shape) > 1:
            jd_embeddings = jd_embeddings.mean(dim=0)

        return jd_embeddings, required_skill_embeddings, role_embeddings

    def match(self, cv_emb, jd_emb, skill_emb, r_skill_emb, exp_emb, role_emb):
        # Cosine similarity between CV and JD
        d_cosine_scores = util.pytorch_cos_sim(cv_emb, jd_emb)
        doc_score = d_cosine_scores.item()

        # Skill similarity
        s_cosine_scores = util.pytorch_cos_sim(skill_emb, r_skill_emb)
        skills_score = s_cosine_scores.item()

        # Experience similarity
        e_cosine_scores = util.pytorch_cos_sim(exp_emb, role_emb)
        experience_score = e_cosine_scores.item()

        skills_score = 0.35 * skills_score
        experience_score = 0.35 * experience_score
        doc_score = 0.3 * doc_score
        total_score = (skills_score + experience_score + doc_score) * 100

        #field_score = (skills_score + experience_score) / 2
        #total_score = ((doc_score + field_score) / 2) * 100

        return {
            "doc_score": doc_score/0.3,
            "skill_score": skills_score/0.35,
            "experience_score": experience_score/0.35,
            #"field_score": field_score,
            "combined_score": total_score,
        }