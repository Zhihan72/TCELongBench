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

class forecastqa_post_verify:
    def __init__(self, question, answer, evidence, day):
        self.question = question
        self.answer = answer
        self.evidence = evidence
        self.day = day
        self.num_coworkers = 3
    
    def compute_quality_bool(self):
        num_coworkers = self.num_coworkers
        answer_q1 = []
        answer_q2 = []
        while (num_coworkers > 0):
            prompt_qual_1 = f"""
            Please verify the question.

            Question Asked: {self.question}
            Correct Answer: {self.answer}
            Evidence: {self.evidence}

            Q1: Do you think the question is clear, unambiguous and detailful?
            A. Yes, the question is clear and unambiguous with detaiful information.
            B. No, the question is ambiguous and may lead to misunderstanding.
            C. I'm not sure/I can't answer/Other

            Q2: Do you think the evidence contains clear and direct information to support the answer to the question?
            A. Yes, the evidence provides clear and direct information to support the answer to the question.
            B. No, the evidence is not able to provide clear and direct information to support the answer to the question.
            C. I'm not sure/I can't answer/Other

            Please output your answer to Q1 and Q2, in the format of "Q1: x\nQ2: x".
            """
            response_qual_1 = ask_ChatGPT(prompt_qual_1)
            resp_list = [item.replace('Q1:','').replace('Q2:','').replace('.','').strip()[0] for item in response_qual_1.split('\n') if len(item.strip())!=0]
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
    
    def compute_drop_bool(self):
        num_coworkers = self.num_coworkers
        answer_q1 = []
        answer_q2 = []
        while (num_coworkers > 0):
            prompt_1 = f"""
            Please verify the question.
            Question Asked: {self.question}
            Note: The above question and its answer come from one article on {self.day}.
            Situation: In order to answer the above question you are given access to all news articles published before {self.day}.
            Task Context: You can imagine going back in time to one day before {self.day}, and on this day you are being posed the question above, while having access to the articles stated in the situation provided.

            Q1: Do you think a person (could be anyone, even an expert in the field) would you be able to make an educated guess as to what the answer to this question is, given the provided situation?
            A. Yes, the person would be able to make an educated guess as to what the answer to this question is.
            B. No, the person would not be able to make an educated guess as to what the answer to this question is.
            C. I'm not sure/I can't answer/Other

            Q2: Do you think a person (could be anyone, even an expert in the field) would be able to find an article (or many) published before {self.day} that answers the question with 100% certainty?
            Note: We don't mean a guess, but rather the article would have a passage that either by itself or with the help of other passages from other articles (all published before {self.day}) would directly answer this question.
            A. Yes, the person would find article(s) from before {self.day} that would directly answer this question.
            B. No, the person would need information from article(s) from {self.day} or after to directly answer this question.
            C. I'm not sure/I can't answer/Other

            Please output your answer to Q1 and Q2, in the format of "Q1: x\nQ2: x".
            """
            response_1 = ask_ChatGPT(prompt_1)
            resp_list = [item.replace('Q1:','').replace('Q2:','').replace('.','').strip()[0] for item in response_1.split('\n') if len(item.strip())!=0]
            if len(resp_list) != 2:
                continue
            answer_q1.append(resp_list[0])
            answer_q2.append(resp_list[1])
            num_coworkers -= 1

        drop_bool = 0
        major_q1 = max(answer_q1, key=answer_q1.count)
        major_q2 = max(answer_q2, key=answer_q2.count)
        if (major_q1 == 'B') or (major_q2 == 'A'):
            drop_bool = 1
        return drop_bool