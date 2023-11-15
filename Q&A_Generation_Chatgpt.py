# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:50:53 2023

@author: scherrmann
"""

import openai
import pandas as pd
import time

# Set your OpenAI API key here
api_key = "##Insert_OpenAI-Key##"

# Initialize the OpenAI API client
openai.api_key = api_key

def generate_question(input_string):
    prompt = f"Create three questions for the following text. It should be possible to answer the question with a substring of the input text. The questions should ask for different aspects of the input. Separate the questions with '\n\n'. The questions should be in German.\n\n Text: {input_string}, \n\n Question:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    question = response.choices[0].message.content.strip()
    return question

def extractive_answer(text, question):
    prompt = f"You have given a text and a question to that text. Find the answer as a substring of the input text. It is crucial that the answer is contained exactly as a substring in the input text. Even if the answer is then not a full sentence or incomplete. Example: Text: 'Herr Müller ist 37 Jahre alt.'\n\n Question: 'Wie alt ist Herr Müller?'\n\n Answer: '37 Jahre'\n\n Text: {text}\n\n Question: {question}\n\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip()
    return answer
    
#%% Load data
data = pd.read_pickle(r"AdHocMultilabel\goldStandard.pkl")
data = data[["Hashs","Sentences","SentenceNr"]]
maxNumSentences = 15
data = data[data["SentenceNr"]<=maxNumSentences] # Cut too long documents
data = data[["Hashs","Sentences"]]
data = data.rename(columns={"Hashs":"id","Sentences":"content"})
data = data.groupby("id")["content"].apply(lambda x : '. '.join(x)).reset_index()
data["content"] = data["content"].str.replace("\.+",".",regex=True) 
#%%

#%% Generate questions 
questions = []
idx = 0
while idx < len(data):
    # Get input from the user
    text = data.content[idx]
    # Generate a suitable question
    try:
        generated_questions = generate_question(text)
    except:
        time.sleep(5)
        continue
    questions.append(generated_questions)

#%% Prepare questions
questionsRaw = pd.Series(questions)
questionsRaw = questionsRaw.str.replace("Antwort.*?(?=\s*\n\s*Question|$)","\n\nQuestion",regex=True)
questionsRaw = questionsRaw.str.replace("Answer.*?(?=\s*\n\s*Question|$)","\n\nQuestion",regex=True)
questionsRaw = questionsRaw.str.replace("Question:*\s*\n*","",regex=True)
questionsRaw = questionsRaw.str.replace("Frage:*\s*\n*","",regex=True)
questionsRaw = questionsRaw.str.replace("\s+$","",regex=True)
questionsRaw = questionsRaw.str.replace("^\s+","",regex=True)
questionsRaw = questionsRaw.str.split("\s*\n+\s*",regex=True)


data_qa = data.copy()
data_qa["question"] = questionsRaw
data_qa = data_qa.explode("question")
data_qa["question"] = data_qa["question"].str.replace("^\s*\d\.?:?\s*","",regex=True)
data_qa["question"] = data_qa["question"].str.strip()
data_qa = data_qa.reset_index(drop=True)

answers = []
idx = 0
while idx < len(data_qa):
    # Get input from the user
    text = data_qa.content[idx]
    question = data_qa.question[idx]
    # Generate a suitable question
    try:
        generated_answer = extractive_answer(text, question)
    except:
        time.sleep(5)
        continue
    answers.append(generated_answer)

answersRaw = pd.Series(answers)
#%% Prepare answers
data_qa["answer"] = answersRaw
data_qa["answerInText"] = data_qa.apply(lambda row: row['answer'] in row['content'], axis=1)

data_qa.to_pickle("Data\\adhoc_data_qa.pkl")
