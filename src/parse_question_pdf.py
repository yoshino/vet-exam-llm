from pdfminer.high_level import extract_text
import re
import pandas as pd


def parse_pdf(pdf_content):
       lines = pdf_content.split("\n")

       for i, line in enumerate(lines):
           if line.startswith('問') or line.startswith('問') or line.startswith('問'):
               start_index = i
               break

       response = []
       content = ""
       for line in lines[start_index:]:
           if line == "":
              continue
       
           if line.startswith('問') or line.startswith('問') or line.startswith('問'):
               if content == "":
                   content += line + "\n"
                   continue
               else:
                   response.append(content)
                   content = line + "\n"
           else:
               content += line + "\n"
       response.append(content)

       return response


def format_questions(response):
    formatted_response = []
    question_no_pattern = r'問\s?\d+'
    
    for res in response:
        lines = res.split("\n")
        answer_choices_start = False
        question = ""
        answer_choices = ""
    
        for i, line in enumerate(lines):
            line = line.replace("", "")
            line = line.replace("〰", "")
            line = line.strip()
            if answer_choices_start:
                answer_choices += line + "\n"
            else:
                question += line
    
            if line.endswith('どれか。') or line.endswith('れか。') or (lines[i-1].endswith('どれ') and lines[i].endswith('か。')):
                answer_choices_start = True
    
        question_no = re.findall(question_no_pattern, question)[0]
        question = question.replace(question_no, "").strip()
        question_no = question_no.replace(" ", "").strip()
        formatted_response.append({"no": question_no, "question": question, "answer_choices": answer_choices.strip()})

    return formatted_response


def convert_question_pdf_to_df(pdf_path):
    pdf_content = extract_text(pdf_path)
    response = parse_pdf(pdf_content)
    formatted_response = format_questions(response)
    return pd.DataFrame(formatted_response)
