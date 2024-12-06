from transformers import pipeline
import torch
from jinja2 import Template
import pandas as pd

def summarize_document(model, prompt, document):
    messages = [
            {"role": "user", "content": prompt.render(text=document)[:8000]},
    ]
    outputs = model(
        messages,
        max_new_tokens=8000,
        do_sample=True,
        temperature=0.01,
        top_k=50,
        top_p=0.95
    )
    return(outputs[0]["generated_text"][-1]["content"])


gemma = pipeline(
    "text-generation",
    model="google/gemma-7b-it",
    model_kwargs={
        "torch_dtype": torch.float16,
        # "quantization_config": {"load_in_8bit": True}
    },
)


arrest_report_prompt = Template("""
Review this police report related to an arrest made by the San Francisco Police Department. Then, provide a concise summary containing key details about the incident. Focus on people's names, locations, dates and times, and descriptions of actions and decisions made by the people described.

------------------

H\nTermination recommended; Resigned\nTerminated prior to IA findings for failure.\nACTION\nOfficer Ornelas resigned on 12/27/2016\ncollateral duty position.\nAppeal in process.\nTermination recommended; Resigned\nOn 3/10/15, demotion to police officer;\nprobation.\nMosqueda resigned on 1/31/19 prior to\nsuspension from hostage negotiator SWAT\nNotice to Terminate served. Resigned\n160-hour suspension, PDSA served\nTerminated 3/18/15 for failure to pass\nremoval from training officer position;\nNotice to Terminate served 7/19/18;\nResigned 10/14/16 prior to the findings.\n9/15/14\n2/24/16:\n5/1/16.\n10/10/2016.\nto pass probation.\nTermination recommended; Officer\nprior to the completion of this case.\n\u20b2\nMar 20 2015\nMar 20 2015\nFinding Dt\nApr 04.2018\nFeb 5 2019\nSep 10 2014\nDec 02.2015\nOct 05 2016\nF\nFinding\nSustained\nSustained\nSustained\nSustained\nSustained Jun 15 2016\nSustained\nSustained |Oct 20 2016\nSustained.\nSustained |Jan 25 2017\nSustained Feb:02-2015\nSustained\nFalsification of Work-Related\nAllegation\nDishonesty\nDishonesty; False Statements\nDocuments; False Statements\nDishonesty; Falsification of Work-\nOn-Duty Sexual Relations\nDocuments\nRelated Documents\nDestruction of Evidence\nFalse Statements\nFalsification of Work-Related\"\nDishonesty\n|On-Duty Sexual Relations\nD\nOfficer Marc Aguilar [1145]\nOfficer Kevin Schindler [1260]\nOfficer Hillary Bjorneboe [1226]\nOfficer Travis Brewer [1132]\nOfficer Jeremy Salcido [1273]\nOfficer Doug Mansker [843]\nDetective Damacio Diaz [854]\nOfficer Manuel Ornelas [989]\nDetective Justin Lewis [1015]\nOfficer Enrique Mosqueda (1242) |Sexual Solicitation\nSr. Officer Kyle Ursery [969):\nC\nOct 06 2017\nOct 16 2014\nOct 16 2014\nFeb 25 2015\nJun 06 2016\nAug 02 2016\nJan 09 2015\nOct 13 2018\nMay 31 2016\nOct 05 2016\nInc Received Dt Involved Officer\nJun 24 2014\nB\nInternal\nInternal\nInternal\nInternal\nInternal\nInternal\nInternal\nInternal\nInternal\nInternal\nType\nInternal\nA\nIA2015-006

------------------

""")

summarize_document(gemma, arrest_report_prompt)