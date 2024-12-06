import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import time 


os.environ["TOKENIZERS_PARALLELISM"] = "true"


document_text = """
<task_description>
As a Legal Clerk, your task is to generate a comprehensive, bulletpoint summary of all the important information contained in the provided document excerpt. Extract all the key details presented in the current page, using the context from the memory log and surrounding pages when necessary for clarity or relevance.
</task_description>

<guidelines>
1. Extract all essential information from the current page.
2. Support the extracted data with additional context from the memory log and surrounding pages to enhance the understanding or relevance of the information in the current page.
3. Use the memory log to help you understand what is relevant and what is irrelevant.
4. DO NOT include any details not explicitly stated in any of the documents.
5. Present the summary in a bullet point format, using subheadings to organize distinct aspects of the information.
6. If the someone's identity is ambiguous, refer to them as "unidentified person". 
7. If some of the information can not be summarized with confidence of its correctness, omit it from your summary.
</guidelines>

<essential_information>
Ensure the summary includes ALL of the following elements, if present. First and foremost, your objective is to return a comprehensive summary that will provide the user with a thorough understanding of the contents of the summary.

Some essential information that will contribute to a comprehensive summary include but are not limited to:
a. Type and purpose of the legal document (e.g., police report, internal investigation)
b. Primary parties involved (full names, roles, badge numbers if applicable)
j. Allegations of misconduct and any associated information
c. Key legal issues, claims, or charges
k. Disciplinary outcomes or their current status
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements) 

For each type of essential information classification, be specific when referring to people, places, and dates. 
</essential_information>

<thinking_process>
Before summarizing, consider:
1. What are the main topics on this page?
2. How does the information relate to previous pages?
3. What context from the memory log is relevant?
</thinking_process>

<output_format>
Present the summary using the following structure:
- Main topic 1
  • Subtopic 1.1
  • Subtopic 1.2
- Main topic 2
  • Subtopic 2.1
  • Subtopic 2.2
</output_format>

<warnings>
- Do not include speculative information
- Avoid summarizing irrelevant details
- Do not draw conclusions not explicitly stated in the text
</warnings>

<reference_materials>
## Current Page ##
Fill In\nOFFICE OF CITIZEN COMPLAINTS - USE BLACK INK ONLY!\nDay, Date &, Time Complaint Received\n1\n10441\n@ 9:30 a.m\n7/30/12\nComplaint Against: Personnel X] Policy | | Procedure L_I\nHow Received: Person 14 Phone || Letter | | SFPD | | Mail-In || Other || : (specify)\nPersonal Information\nO Co-Complainant\nPrimary Complainant:\n3\n2\nAge: 23 Date of Birth\nSex: Female\nMiddle Initial\nLast Name\nFirst Name\nEthnicity: BIK/samoan\nDesk Clerk\nHOME ADDRESS\nOccupation:\nApartment\nStreet\n4\nTelephone Numbers:\nStat\nCity\nZip\nHome:\nWORK ADDRESS:\nApartment\nStreet\nWork:\nState\nCity\nZip\nType of Place\nDistrict\nLocation of Occurrence:\n7\n5\n04 - Residence\n3C. Baynew\nA.M. / P.M.\nIncident Report or Citation No\nDay, Date, & Time Of Occurrence:\n8\n(Circle one)\nTues, 7/17/2012\n1538\n10\nSECONDARY COMPLAINANT? Yes| | No) Witnesses? Yes / No | | (If \"Yes\", attach separate sheet of paper.)\nTaped Interview? Yes / No [] Criminal Case Pending in Relation to this matter? Yes I NON\nInjuries Claimed? Yes M4 No | | Injuries Visible? Yes / No | | Drug/Alcohol Related? Yes | |\nNo LI\n11\nCOMP\nOther:\nPhotos Taken? Yes M No [ By: Photo Lab 11 O.C.C. 11\nType of Injury: LACERATION TO CHEST (BOYFRIEND)\nMedical Release Signed? Yes | | No |_|\n14\n15\n13\n18\n19\n20\n17\n16\n12\n21\nMember's Name & Star Number\nUnit\nDISP.\nSvc\nType\nUniform\nRank\nSex\nActivity\nEth\nYes No\nSEE ALLEGATION\nCONTINUATION FORM\nSF DPA - 0441-12 - 000001
0441-12\nOn July 17th 2012 Tuesday I was at home with my boyfriend named\nand My Dog. We\nwere both getting ready to start our day; I was on my way to work. While we were both getting ready,\nwe can both hear outside a lot of commotion going on in the parking lot. We both stuck our heads out\nthe window, to only see several Police officials banging on my door. I was immediately frightened by the\nPolice aggressiveness on forcing themselves into my house, and threaten me to open the door. I\nasked, \"What did they want?\" Officer Sanders replied, \"Someone they were chasing ran in my house,\nOpen the Door or else we're going to kick it down!\" I assure them that no one entered my house at all,\nand that if they didn't have a search warrant they could not enter, one of the Officers said,\"they had one\nit was on its way\". I didn't have on any clothes and try to grab something to cover myself so that I could\nhurry to get to the door. When I opened the door, Officer Sanders kept yelling\nI told\nhim no one by that name lives here nor did anyone by that name enter my house. Within seconds I was\npulled out by several officers in my Panties and my Bra, and they rushed in. My boyfriend was already\ncoming down the stairs to figure out what was going on as well, he was also trying to get dress. Before\nhe had a chance to say who he was the police grabbed him swung him around the stairs, he was\nslammed on top of kitchen table where there was a glass cup at, and the entire cup was broken into his\nchest. I screamed and cried out for them to stop! They were punching him and forcing his arms to his\nback. He didn't do anything that would make the police feel threaten, I even seen a police officer in my\nliving room draw his gun down on my boyfriend, While he was on the ground cuffed. I asked the police\nofficers why they were doing this. Four officers guarded my front door preventing me to enter my house\nto at least put some clothes on; my neighbor came down to give me some sweat pants. 16 SFPD officers\nfor a suspect they thought ran into my house.\nsearched my house\nWhen they realized he was never in my house. SFPD cont. to treat us like criminals they finally asked us\nboth our identity and ran our names. My boyfriend was on the kitchen floor bleeding out; He was cut\nreally bad with a lot of glass in his chest. The SFPD called the Ambulance for medical treatment for him.\nAfter the officers ran our names they cited us off, and wrote my boyfriend a ticket for resisting, which is\nnot true. The officers were getting ready to leave and left by telling me this was just one big\nmisunderstanding.\nI couldn't believe what just happened to me and my boyfriend. I was scared .. My boyfriend sat in San\nFrancisco General Hospital for 6hours. They had to give him a CT scan to make sure his Jaw wasn't\nbroken because it was swollen and he couldn't bite down. The doctor also requested for him to have\nan X-ray exam to make sure he didn't have any broken glass inside his chest. When we got to the final\nhis diagnosis was a laceration to his chest wall, He was giving several stitches\nstages of treating\nand prescribed pain medication because of the pain he was in. He did not deserve this nor did myself. I\nwas Humiliated outside in front of all my neighbors, exposing myself to the public. I am scared to look at\nanyone I know that seen me outside like that, it is very embarrassing and just makes me feel shameful\nof myself. I don't agree with the SFPD actions, and I want to file a complaint against all the Officers who\nentered my home. The following will list the Names and Badge numbers:\nF.Abi Chine #4251\nDejesus.L #247\nSF DPA - 0441-12 - 000005
</reference_materials>

<output_instruction>
Generate the current page summary below:
</output_instruction>
"""


# Create messages list with the document analysis task
messages = [
    # {"role": "system", "content": "You are a helpful AI assistant skilled at analyzing documents and extracting information."},
    {"role": "user", "content": document_text}
]

# Load the model and tokenizer
model_name = "microsoft/Phi-3.5-mini-instruct" # tied for 2nd best
# model_name = "google/gemma-2-2b-it" # tied for best
# model_name = "Qwen/Qwen2.5-3B-Instruct" # fail
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # best
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="mps"
)

# Create a pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Generation configuration
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.7,
    "do_sample": True,
    "top_k": 10,
    "top_p": 0.95
}

start_time = time.time()
output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")