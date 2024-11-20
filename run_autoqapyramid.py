from typing import List
import json
import datasets
import nltk
from tqdm import tqdm
import openai
import os

class GPT4o:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    
    def format_prompt(self, prompt):
        return [{"role": "user", "content": prompt}]
    
    def run(self, prompt):
        if isinstance(prompt, str):
            prompt = self.format_prompt(prompt)
        
        response = self.client.chat.completions.create(
                model=self.model_id,
                messages=prompt
        )
        
        response = response.choices[0].message.content

        return response
    
    def postprocess_response(self, response):
        return {"role": "assistant", "content": response}

prescence_prompt = gen_prompt = """Read the following summary. Then read a question and an answer. Answer whether the question and answer pair can be inferred from the summary. Please strictly output either [YES] or [NO].

[Summary]
[SUMMARY]

[Question]
[QUESTION]

[Answer]
[ANSWER]"""

class AutoQAPyramid:
    def __init__(self, data=None):
        self.model = GPT4o()

    def run_qa_gen(self, reference):
        print("Running QA Generation with QASem")
        from qasem_parser import QasemParser, QasemFrame
        arg_parser_path = "cattana/flan-t5-xl-qasem-joint-tokenized"
        parser = QasemParser.from_pretrained(arg_parser_path)
        
        reference_sent = []
        for ref in reference:
            reference_sent.extend(nltk.sent_tokenize(ref))

        reference_sent = list(set(reference_sent))
        frames = parser(reference_sent)
        sent2frame = dict()
        for sent, frame in zip(reference_sent, frames):
            frame = self.parse_qasem(frame)
            sent2frame[sent] = frame

        ref2qa = dict()

        qas = []
        for ref in reference:
            if ref not in ref2qa:
                ref_sent = nltk.sent_tokenize(ref)
                qa = []
                for r in ref_sent:
                    qa.extend(sent2frame[r])
                
                ref2qa[ref] = qa
            else:
                qa = ref2qa[ref]
            qas.append(qa)
        return qas

    def parse_qasem(self, pred):
        pred = [str(p) for p in pred]
        predictions = list()
        for p in pred:
            k = p[:p.index(":")-2]
            v = p[p.index(":")+1:].strip()
            for x in v.split("|"):
                # print(p)

                answer = x[:x.index("(")].strip()
                question = x[x.index("("):-1].strip()

                if ":" in question:
                    question = question[question.index(":")+1:].strip()
                
                if "(" in question:
                    question = question[question.index("(")+1:].strip()
                if ")" in question:
                    question = question[:question.index(")")].strip()

                predictions.append( (k, question, answer) )
        return predictions
    
    def run_qa_presence(self, qas, summary):
        print("Running QA Presence")
        scores = []
        for qa, summ in tqdm(zip(qas, summary), total=len(qas)):
            score = []
            for verb, q, a in tqdm(qa, leave=False):
                cur_prompt = prescence_prompt.replace("[SUMMARY]", summ).replace("[QUESTION]", q).replace("[ANSWER]", a)
                output = self.model.run(cur_prompt)
                # extract score
                if output == "[YES]":
                    score.append(1)
                elif output == "[NO]":
                    score.append(0)
                else:
                    if "yes" in output.lower():
                        score.append(1)
                    elif "no" in output.lower():
                        score.append(0)
                    else:
                        score.append(None)
            scores.append(score)
        return scores
    
    def run(self, summaries: List[str], references: List[str] = None, qas: List[List[str]] = None):
        assert references is not None or qas is not None, "Either references or qas must be provided"

        if qas is None:
            qas = self.run_qa_gen(references)
        
        scores = self.run_qa_presence(qas, summaries)

        return qas, scores


dataset = datasets.load_dataset("shiyue/QAPyramid")["cnndm_test"]

ids, system, reference, summary = [], [], [], []
qas = []

for dat in dataset:
    ref = dat["reference"]
    summaries = dat["system_outputs"]
    for k, v in summaries.items():
        if v is not None:
            ids.append(dat["example_id"])
            system.append(k)
            reference.append(ref)
            summary.append(v)
            qas.append(dat["QAs"])

autoqapyramid = AutoQAPyramid()
qas, scores = autoqapyramid.run(summary, references=reference)

with open("autoqapyramid_qas.json", "w") as f:
    json.dump(qas, f, indent=4)

with open("autoqapyramid_scores.json", "w") as f:
    json.dump(scores, f, indent=4)