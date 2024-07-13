import os
import openai

openai.api_key = ""
model_engine = "gpt-3.5-turbo-instruct"
def ask_ChatGPT(msg):
    completion = openai.Completion.create(
        model=model_engine,
        prompt=msg,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    response = completion.choices[0].text
    return response

class orderqa_post_verify:
    def __init__(self, points):
        self.points = points
        self.num_coworkers = 3
    
    def compute_quality_bool(self):
        num_coworkers = self.num_coworkers
        answer_q1 = []
        answer_q2 = []
        answer_q3 = []
        answer_q4 = []
        while (num_coworkers > 0):

            prompt_qual_1 = f"""
            Below are key points presenting a storyline. Please verify this storyline.
            {self.points}

            Q1: Do you think the above key points are arranged in a chronological order?
            A. Yes, the above key points are apparently arranged in a chronological order.
            B. No, swapping some of them can make the storyline more chronological.
            C. I'm not sure/I can't answer/Other

            Q2: Do you think each of the above key points represents a event that just happened or is happening?
            A. Yes, they all represent the events that just happened or is happening.
            B. No, some of them discuss the static content of certain documents, someone's view or events that may happen in the future and/or happened before.
            C. I'm not sure/I can't answer/Other
            
            Please output your answer to Q1 and Q2, in the format of "Q1: x\nQ2: x".
            """
            response_qual_1 = ask_ChatGPT(prompt_qual_1)
            resp_list = [item.replace('Q1:','').replace('Q2:','').replace('Q3:','').replace('Q4:','').replace('.','').strip()[0] for item in response_qual_1.split('\n') if len(item.strip())!=0]
            if len(resp_list) != 2:
                continue
            answer_q1.append(resp_list[0])
            answer_q2.append(resp_list[1])
            num_coworkers -= 1

        qual_bool = 0
        major_qual_q1 = max(answer_q1, key=answer_q1.count)
        major_qual_q2 = max(answer_q2, key=answer_q2.count)
        if (major_qual_q1=='A') and (major_qual_q2=='A'):
            qual_bool = 1
        return qual_bool