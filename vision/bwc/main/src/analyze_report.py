import os
import logging
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
import multiprocessing
from collections import namedtuple
from functools import partial
import concurrent.futures
from datetime import datetime
import time
import re
from typing import List
from langchain_openai import ChatOpenAI
import sys

Doc = namedtuple("Doc", ["page_content", "metadata"])

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


def setup_llm():
    return ChatOpenAI(model="gpt-4o-mini")

llm = setup_llm()

memory_log_template = """
As a Legal Clerk, your task is to review the new summary and update the memory log only when the new summary contains crucial information directly related to the main subject of the document. Maintain a concise memory log that focuses on the key aspects of the events, allegations, investigations, and outcomes described in the document.
</task_description>

<guidelines>
1. Review and Compare:
   • Carefully review the current memory log and the new summary.
   • Determine if the new summary contains crucial information that is not already in the memory log.

2. Identify Crucial Information:
   • Focus on information specific to the main subject of the document.
   • Look for key details related to events, allegations, investigations, and outcomes.

3. Update Selectively:
   • Only update the memory log if the new summary contains crucial information not already present.
   • If updating, integrate the new information seamlessly into the existing log.

4. Maintain Conciseness:
   • Keep the memory log focused and concise.
   • Avoid redundancy or unnecessary details.

5. Ensure Accuracy:
   • Only include information that is directly stated in the document.
   • Do not infer or speculate beyond what is explicitly mentioned.

6. Preserve Original Structure:
   • If no update is necessary, reproduce the original memory log without changes.
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
Before updating the memory log, consider:
1. Does the new summary contain any crucial information not already in the memory log?
2. How does this new information relate to the main subject of the document?
3. Can this new information be integrated into the existing log without disrupting its flow?
4. Is this information essential to understanding the key aspects of the case?
5. Am I maintaining the conciseness of the log while including all crucial details?
</thinking_process>

<warnings>
- Do not add information that is not directly stated in the document
- Avoid speculation or inference beyond what is explicitly mentioned
- Do not remove or alter existing crucial information in the memory log
- Ensure that any updates maintain the chronological and logical flow of events
- Be cautious of potential inconsistencies between the new summary and existing log
</warnings>

<reference_materials>
## Original Memory Log ##
{memory_log}

## New Summary ##
{summary}
</reference_materials>

<output_instruction>
Based on your review of the current memory log and the new summary, provide either an updated memory log incorporating the crucial new information, or reproduce the original memory log if no update is necessary. Ensure the output maintains a concise focus on key aspects of events, allegations, investigations, and outcomes related to the main subject of the document:
</output_instruction>
"""

summary_template_for_memory_log = """
<document_classification>
First, determine if this document is a legal document or another document type. Consider the following:
- Does it contain legal terminology, case numbers, or references to laws and regulations?
- Is it structured like a legal document (e.g., contracts, court filings, police reports)?
- Does it discuss legal proceedings, rights, or obligations?

Based on your analysis, classify this document as either:
1. Legal Document
2. Other Document Type
</document_classification>

<task_description>
Your task is to generate a comprehensive, bulletpoint summary of all the important information contained in the provided document excerpt. Extract all the key details presented in the current page.
</task_description>


<legal_document_essential_information>
If the document is classified as a legal document, ensure the summary includes ALL of the following elements, if present:

b. Primary parties involved (full names, roles, badge numbers if applicable)
c. Key legal issues, claims, or charges
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
j. Allegations of misconduct and any associated information
k. Disciplinary outcomes or their current status
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements) 
For each type of essential information, be specific when referring to people, places, and dates. 
</legal_document_essential_information>


<other_document_type_guidelines>
If classified as Other Document Type, follow these guidelines:

Ensure the summary includes the following elements, if present:
b. Main topics or themes
c. Key individuals or organizations mentioned
d. Important dates, events, or milestones
e. Significant data or findings
f. Main arguments or conclusions
g. Any recommendations or future actions proposed
h. Relevant background information or context

</other_document_type_guidelines>

<thinking_process>
Before summarizing, consider:
1. Is this a legal document or another document type?
2. What are the main topics on this page?
3. What are the most important pieces of information to extract based on the document type?
</thinking_process>

<output_format>
Present the summary using the following structure:

- Main events
  • Sub-event 2.1
  • Sub-event 2.2

- Main actions
  • Sub-action 4.1
  • Sub-action 4.2

- Main legal issue
  • Sub-legal ossue 5.1
  • Sub-legal issue 5.2

- Main legal procedure
  • Sub-legal procedure 6.1
  • Sub-legal procedure 6.2

- Main allegations 
  • Sub-allegation 7.1
  • Sub-allegation  7.2

- Main disiplinary outcomes
  • Sub-disciplinary outcome 8.1
  • Sub-disciplinary outcome 8.2

</output_format>

<warnings>
- Do not include speculative information
- Avoid summarizing irrelevant details
- Do not draw conclusions not explicitly stated in the text
</warnings>

<reference_materials>
Current Page:
{current_page}
</reference_materials>

<output_instruction>
First, state the document classification (Legal Document or Other Document Type) and provide a brief explanation for your decision. Then, generate the current page summary following the appropriate guidelines based on the classification.
</output_instruction>
"""


page_summary_template = """
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
{current_page}
</reference_materials>

<output_instruction>
Generate the current page summary below:
</output_instruction>
"""


page_summary_verification_template = """
As a Legal Document Verifier, your task is to review the original document and its summary, then produce an enhanced summary that incorporates important missing information and corrects any inaccuracies.

<task description>
Guidelines:
1. Carefully review the original document and its existing summary.
2. Identify important information in the original document that is missing from the summary.
3. Check for any inaccuracies or misrepresentations in the summary.
4. Create an enhanced summary by:
   a. Adding missing important information
   b. Correcting any inaccuracies
   c. Preserving accurate existing content
5. Ensure that the enhanced summary is comprehensive, accurate, and relevant.
6. Maintain the original structure and flow of the summary as much as possible.
7. If the someone's identity is ambiguous, refer to them as "unidentified person". 
8. If no significant changes are needed, return the original summary with a note stating it's accurate and complete.
</task description>

<thinking_process>
Before enhancing the summary, consider:
1. What important information is present in the original document but missing from the current summary?
2. Are there any inaccuracies or misrepresentations in the summary compared to the original document?
3. Where in the existing summary structure does new information or corrections best fit?
4. How can I integrate this information smoothly without disrupting the existing flow?
5. Does this new information or correction provide additional context or clarity to the existing summary?
</thinking_process>

<output_format>
Present the enhanced summary using the existing structure of the current summary. Add new information or corrections where appropriate. For example:

Enhanced Summary:
- Main topic 1
  • [UNCHANGED] Sub-topic 1.1
  • [UNCHANGED] Sub-topic 1.2

- Main topic 2
  • [UNCHANGED] Sub-topic 2.1
  • [CORRECTED] Sub-topic 2.2

- Main topic 3
  • [UNCHANGED] Sub-topic 3.1
  • [ADDED] Sub-topic 3.2

Maintain this format throughout the summary, inserting new information or corrections where they fit best within the existing structure.
</output_format>

<reference documents>
Original Document:
{original_document}

Current Summary:
{current_summary}
</reference documents>

Please provide the enhanced summary below, with new information and corrections clearly marked, or the original summary if no changes are needed. 

Enhanced Summary or Original Summary:
"""


combine_template = """
<task_description>
As a Legal Clerk, your task is to concatenate the provided page summaries into a single, comprehensive, and well-organized summary for the given section of the police report. Your goal is to create the best possible summary by taking the most important and relevant information from each provided summary and combining them into a detailed, chronological, and coherent summary without any duplication.
</task_description>

<guidelines>
1. Comprehensive Information Integration:
   • Review the current combined summary and the new page summaries to extract the most important information that is relevant to producing a summary.

2. Narrative Coherence:
   • Support the extracted data with additional context from the memory log and surrounding pages to enhance the understanding or relevance of the information in the current page.
   • Use the memory log to help you understand what is relevant and what is irrelevant.

3. Handling Contradictions:
   • If inconsistencies arise between the summaries, prioritize the most detailed and specific information.
   • If the information is incomplete, do not include it.

4. Factual Accuracy:
   • DO NOT include any details not explicitly stated in either summary.

5. Formatting for Clarity:
   • Ensure that the updated combined summary is formatted as bullet points with a logical flow of information.
   • If possible, organize the bullet points chronologically.
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
Before combining the summaries, consider:
1. What are the main topics covered across all summaries?
2. How can the information be organized chronologically?
3. Are there any contradictions or inconsistencies between summaries?
4. What information from the memory log provides crucial context?
5. How can I ensure all essential information is included without duplication?
</thinking_process>

<warnings>
- Do not include speculative information
- Avoid summarizing irrelevant details
- Do not draw conclusions not explicitly stated in the summaries
- Do not omit critical information even if it appears in multiple summaries
- Ensure that all information is accurately attributed to the correct parties and events
</warnings>

<reference_materials>
## Summary 1 ## 
{summary_1}

## Summary 2 ## 

{summary_2}
</reference_materials>

<output_instruction>
Generate the updated combined summary below, ensuring it adheres to all guidelines, includes all essential information, and is presented in a clear, bullet-point format:
</output_instruction>
"""

verification_template = """
<task_description>
Your task is to meticulously review the combined summary, which integrates content from two individual summaries (summary 1 and summary 2) of a document. This verification process aims to ensure that:
1. ONLY the most crucial information from both original summaries is included in the combined summary.
2. Each section adheres strictly to the 6 bulletpoint limit.
4. The essence of the original document is effectively captured within these constraints.
</task_description>

<verification_guidelines>
1. Critical Information Check:
   • Identify the most important points from both original summaries.
   • Verify that only the most crucial information is included in the combined summary.

2. Bulletpoint Limit Verification:
   • Confirm that each section contains EXACTLY 6 bulletpoints, no more and no less.
   • If any section doesn't meet this requirement, flag it for revision.

3. Information Integration Check:
   • Identify any information that differs between the original summaries.
   • Verify that complementary information (e.g., names in one summary, roles in another) is properly merged into comprehensive bulletpoints.

4. Information Accuracy:
   • Ensure that the critical details from both summaries are accurately represented.
   • Verify that no speculative information or unwarranted conclusions have been added.

5. Missing Crucial Information:
   • If any critical information is missing, evaluate its importance against the existing points.
   • If necessary, recommend specific replacements to include missing crucial points while maintaining the 6 bulletpoint limit.

6. Clarity and Conciseness:
   • Ensure each bulletpoint is clear, concise, and focuses on a single main idea or closely related ideas.
   • Verify that the language used is precise and unambiguous.
</verification_guidelines>


<legal_document_essential_information>
For legal documents, verify that the most crucial elements from the following list are included, if present in the original summaries:
a. Primary parties involved (full names, roles, badge numbers if applicable)
b. Key legal issues, claims, or charges
c. Critical events or incidents (with specific dates, times and locations)
d. Main findings or decisions
e. Significant evidence or testimonies
f. Important outcomes or rulings
g. Current status of the matter
h. Any pending actions or future proceedings
i. Allegations of misconduct and any associated information
j. Disciplinary outcomes or their current status
k. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements)

Ensure only the most crucial elements are included within the 6 bulletpoint limit per section.
</legal_document_essential_information>

<thinking_process>
During the verification process, consider:
1. What are the absolute most crucial points from both original summaries?
2. Does each section contain exactly 6 bulletpoints of critical information?
4. Is the essence of the document effectively captured within these constraints?
5. Are there any crucial details missing that should replace less important information?
6. Is each bulletpoint clear, concise, and focused on a single main idea or closely related ideas?
7. Have I maintained the original meaning and context of the information?
</thinking_process>

<warnings>
- Include only the most critical information within the 6 bulletpoint limit per section.
- Do not alter the meaning or context of any information during the verification process.
- Maintain any crucial ambiguities or uncertainties from the original summaries.
- Do not add speculative information or draw conclusions not present in the original summaries.
- If recommending changes, provide specific suggestions that maintain the bulletpoint limit.
- Prioritize accuracy and crucial information over minor details.
</warnings>

<output_format>
Verify that the combined summary follows this format: 

**IMPORTANT: Each section MUST contain EXACTLY 6 bulletpoints.**

- Legal Issues, Claims, and Charges (EXACTLY 6 BULLETPOINTS)
  • Bulletpoint 1 
  • Bulletpoint 2 
  • Bulletpoint 3 
  • Bulletpoint 4
  • Bulletpoint 5
  • Bulletpoint 6

- Key Events and Incidents (EXACTLY 6 BULLETPOINTS)
  • Bulletpoint 1 
  • Bulletpoint 2 
  • Bulletpoint 3 
  • Bulletpoint 4
  • Bulletpoint 5
  • Bulletpoint 6

- Main Findings, Decisions and Actions (EXACTLY 6 BULLETPOINTS)
  • Bulletpoint 1 
  • Bulletpoint 2 
  • Bulletpoint 3 
  • Bulletpoint 4
  • Bulletpoint 5
  • Bulletpoint 6

</output_format>

<reference_materials>
Current Combined Summary:
{current_combined_summary}

Summary 1:
{summary_1}

Summary 2:
{summary_2}
</reference_materials>

<output_instruction>
1. Verify the Current Combined Summary against the original summaries using the guidelines provided.
2. If changes are necessary:
   a. Provide specific recommendations for each change.
   b. Ensure all changes maintain the 5 bulletpoint limit per section.
   c. Present your recommendations clearly, indicating which bulletpoints should be replaced or modified.
5. Present the final Verified Summary in a clear, bullet-point format, organized chronologically for legal documents or thematically for other document types.
6. Do not include any reference to your verification process in the final output.

## Verified Summary: ##
</output_instruction>
"""

condense_template = """
<task_description>
As a Legal Clerk, your task is to condense the summaries into a single summary.
</task_description>

<essential_information>
Ensure the condensed summary includes ALL of the following elements (if present in the summary). First and foremost, your objective is to return a comprehensive summary that will provide the user with a thorough understanding of the contents of the summaries that you are condensing. 

Some essential information that will contribute to a comprehensive summary include but are not limited to:
b. Primary parties involved (full names, roles, badge numbers if applicable)
j. Allegations of misconduct and any associated information
c. Key legal issues, claims, charges, or arguments
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
Before condensing the summary, consider:
1. What are the most critical pieces of information that must be retained?
2. How can I organize the information to present a clear summary?
4. How can I ensure that the condensed summary remains coherent and comprehensive?
5. Are there any redundancies in the merged summary that can be eliminated?
</thinking_process>

<warnings>
- Do not introduce new information not present in the merged summary
- Avoid altering the meaning or context of any information during the condensing process
- Do not omit any essential details, even if struggling to meet the 5-paragraph limit
- Ensure that all information remains accurately attributed to the correct parties and events
- Be cautious of potential inconsistencies and address them appropriately
- Do not include speculative or inferential information
</warnings>

<reference_materials>
## Input Summary ##
{summaries}
</reference_materials>

<output_instruction>
Provide the condensed summary below, ensuring that all essential information from the merged summary is retained, accurately presented, and organized in a clear, chronological, and logical manner. The condensed summary should not exceed 5 paragraphs:
</output_instruction>
"""


final_combine_template = """
<task_description>
Your task is to combine the provided summaries into a single, comprehensive, and well-organized final summary for the given document. Your primary goal is to preserve ALL important information from both summaries, creating a comprehensive summary without any omissions.
</task_description>

<document_type_check>
First, determine if the summaries are from a legal document or another document type based on the classification provided in each summary. State the document type before proceeding with the combination.
</document_type_check>

<guidelines>
1. Comprehensive Information Integration:
   • Include ALL important information from both summaries, even if it results in a longer combined summary.
   • Use bullet points to capture all important details. 

3. Factual Accuracy:
   • Include only details explicitly stated in either summary.
   • If information is incomplete or unclear, maintain that ambiguity rather than making assumptions.

4. Completeness Check:
   • After combining, review both original summaries to ensure no important information has been omitted.
   • If any omissions are found, immediately add them to the combined summary with appropriate source tags.
</guidelines>

<legal_document_essential_information>
If the document is classified as a legal document, ensure the summary includes ALL of the following elements, if present:
b. Primary parties involved (full names, roles, badge numbers if applicable)
c. Key legal issues, claims, charges, or arguments
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
j. Allegations of misconduct and any associated information
k. Disciplinary outcomes or their current status
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements) 

For each type of essential information, be specific when referring to people, places, and dates. 
</legal_document_essential_information>

<other_document_type_essential_information>
If the document is classified as another document type, ensure the summary includes the following elements, if present:
b. Main topics and themes
c. Key individuals and/or organizations mentioned
d. Important dates, events, or milestones
e. Significant data, statistics, or findings
f. Main arguments and conclusions
g. Any recommendations or future actions proposed
h. Relevant background information or context
</other_document_type_essential_information>

<thinking_process>
Before and during the combination of summaries, consider:
1. Is this a legal document or another document type?
2. What are the main topics covered across all summaries?
3. Are there any contradictions or inconsistencies between summaries that need to be highlighted?
4. What information provides crucial context, and how can I ensure it's not lost in the combination?
5. Have I double-checked that no important information from either summary has been omitted?
</thinking_process>

<warnings>
- Prioritize completeness. It's better to include all important information than to omit details for brevity.
- Do not include speculative information or draw conclusions not explicitly stated in the summaries.
- Do not alter the meaning or context of any information when integrating it into the combined summary.
</warnings>

<reference_materials>
Summary 1:
{summary_1}

Summary 2:
{summary_2}
</reference_materials>

<output_instruction>
Generate the final combined summary below, ensuring it adheres to all guidelines, includes all essential information based on the document type, and is presented in a clear, bullet-point format:

## Updated Combined Summary: ##
</output_instruction>
"""


final_verification_template = """
<task_description>
Your task is to meticulously review the combined summary, which integrates content from two individual summaries (summary 1 and summary 2) of a document. This verification process aims to ensure that ALL relevant information from both original summaries is accurately contained within the combined summary, including key details, events, findings, and outcomes related to the document from both sources. Include relevant information from the memory log if it improves the overall summary.
</task_description>

<document_type_check>
First, confirm the document type (Legal Document or Other Document Type) based on the classification provided in the combined summary. State the document type before proceeding with the verification.
</document_type_check>

<verification_guidelines>
1. Systematic Comparison:
   • Create a checklist of all important points from both original summaries.
   • Systematically check each point against the combined summary, marking items as present or missing.

2. Information Preservation:
   • Ensure that ALL important details from both summaries are accurately incorporated into the combined summary.

3. Missing Information Addition:
   • For any information found missing during the review, explicitly add it to the verified summary. 
</verification_guidelines>

<legal_document_essential_information>
If the document is classified as a legal document, ensure the summary includes ALL of the following elements, if present:
b. Primary parties involved (full names, roles, badge numbers if applicable)
c. Key legal issues, claims, charges, or arguments
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
j. Allegations of misconduct and any associated information
k. Disciplinary outcomes or their current status
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements) 

For each type of essential information, be specific when referring to people, places, and dates. 
</legal_document_essential_information>

<other_document_type_essential_information>
If the document is classified as another document type, ensure the summary includes the following elements, if present:
b. Main topics or themes
c. Key individuals or organizations mentioned
d. Important dates, events, or milestones
e. Significant data, statistics, or findings
f. Main arguments or conclusions
g. Any recommendations or future actions proposed
h. Relevant background information or context
</other_document_type_essential_information>

<thinking_process>
During the verification process, consider:
1. Have I created a comprehensive checklist of all important points from both original summaries?
2. Am I systematically comparing each point to ensure nothing is missed?
3. After my initial review, did I re-read the original summaries to catch any overlooked details?
4. If I found any missing information, have I added it to the verified summary with proper source attribution?
</thinking_process>

<warnings>
- Prioritize completeness over conciseness. 
- Do not alter the meaning or context of any information during the verification process.
- Ensure that all ambiguities or uncertainties from the original summaries are maintained.
- Do not add speculative information or draw conclusions not present in the original summaries or combined summary.
- If significant information is missing or inaccurately represented, be prepared to recommend a re-combination of the summaries.
</warnings>


<reference_materials>
Current Combined Summary:
{current_combined_summary}

Summary 1:
{summary_1}

Summary 2:
{summary_2}
</reference_materials>


<output_format>
Provide the verified summary below. If no changes are needed, return the contents of the Updated Summary. Present the summary in a clear, bullet-point format, organized chronologically, if relevant. Do not include any reference to your verification check in the final output.

## Verified Summary ##
- Main topic 1
  • [UNCHANGED] Sub-topic 1.1
  • [UNCHANGED] Sub-topic 1.2
  • [UNCHANGED] Sub-topic 1.4
  • [CORRECTED] Sub-topic 1.3

- Main topic 2
  • [UNCHANGED] Sub-topic 2.1
  • [UNCHANGED] Sub-topic 2.2
  • [UNCHANGED] Sub-topic 2.3
  • [ADDED] Sub-topic 2.4

- Main topic 3
  • [UNCHANGED] Sub-topic 3.1
  • [UNCHANGED] Sub-topic 3.2
  • [UNCHANGED] Sub-topic 3.3
  • [UNCHANGED] Sub-topic 3.4

</output_format>
"""

improve_summary_template = """
As a Legal Clerk, your task is to enhance the existing summary of a legal document by incorporating important information from the memory log. The current summary contains specific details, while the memory log provides high-level, important information. 
Your goal is to improve the summary by adding only the most important missing information from the memory log without removing any existing content.

<task description>
Guidelines:
1. Carefully review both the current summary and the memory log.
2. Identify the most information in the memory log that is not present in the current summary.
3. Add the missing important information to the appropriate sections of the summary.
4. Preserve all existing content in the current summary.
5. Ensure that added information enhances the summary's comprehensiveness and relevance.
6. Maintain the original structure and flow of the summary as much as possible.
7. If no significant additions are needed, return the original summary unchanged.
</task description>

<essential information>
When reviewing the memory log, pay special attention to these elements if they're not already in the summary:
a. Primary parties involved (full names, roles, badge numbers if applicable)
b. Key legal issues, claims, charges, or arguments not mentioned in the summary
c. Critical events or incidents (with specific dates, times, and locations)
d. Main findings or decisions not covered in the summary
e. Significant evidence or testimonies missing from the summary
f. Important outcomes or rulings not mentioned
g. Updates to the current status of the matter
h. Any pending actions or future proceedings not included
i. Allegations of misconduct and any associated information
j. Disciplinary outcomes or their current status
k. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements)
For any new information added, be specific when referring to people, places, and dates.
</essential information>

<thinking_process>
Before adding to the summary, consider:
1. What important information is present in the memory log but missing from the current summary?
2. Where in the existing summary structure does this new information best fit?
3. How can I integrate this information smoothly without disrupting the existing flow?
4. Does this new information provide additional context or clarity to the existing summary?
</thinking_process>

<output_format>
Present the enhanced summary using the existing structure of the current summary. Add new information from the memory log where appropriate, clearly marking additions with [ADDED] tags. For example:

Enhanced section:
- Main topic 1
  • [UNCHANGED] Sub-topic 1.1
  • [UNCHANGED] Sub-topic 1.2
  • [UNCHANGED] Sub-topic 1.4
  • [CORRECTED] Sub-topic 1.3

- Main topic 2
  • [UNCHANGED] Sub-topic 2.1
  • [UNCHANGED] Sub-topic 2.2
  • [UNCHANGED] Sub-topic 2.3
  • [ADDED] Sub-topic 2.4

- Main topic 3
  • [UNCHANGED] Sub-topic 3.1
  • [UNCHANGED] Sub-topic 3.2
  • [UNCHANGED] Sub-topic 3.3
  • [UNCHANGED] Sub-topic 3.4

Maintain this format throughout the summary, inserting new information where it fits best within the existing structure.
</output_format>

<reference documents>
Current Summary:
{summary}

Memory Log:
{memory_log}
</reference documents>

Please provide the enhanced summary below, with new information clearly marked, or the original summary if no changes are needed. 

Enhanced Summary or Original Summary:
"""

organization_template = """
<task_description>
As a Legal Clerk, your task is to reorganize the content from concatenated summaries into a single, coherent, and condensed summary. If possible, the summary should cover the details of the document chronologically from the document's beginning to its end, temporally.
</task_description>

<essential_information>
Ensure the reorganized summary includes all important elements from the concatenated summaries that are necessary to providing a complete summary. Your objective is to provide a thorough understanding of the case without redundancies. Focus on:
a. Unique parties involved (with full details where available)
b. Distinct allegations of misconduct
c. Unique legal issues, claims, or charges
d. Critical events or incidents (without repeating duplicate entries)
e. Findings or decisions (consolidating where appropriate)
f. Fignificant evidence or testimonies (without repetition)
g. Unique outcomes, rulings, or disciplinary actions
h. Current status of all aspects of the case
i. Any unique pending actions or future proceedings
j. Procedural events
</essential_information>

<relevance_and_coherence>
When organizing the information:
1. Evaluate the importance of each piece of information in relation to the case's outcome or current status.
2. Include only the most crucial events, issues, and decisions.
3. Eliminate redundant or less significant information.
4. Ensure each point contributes significantly to understanding the case's core issues and resolution.
5. Maintain a clear narrative flow that shows the relationship between different aspects of the case.
6. Prioritize the most recent or final decisions and outcomes when applicable.
7. Consider the temporal sequence of events, emphasizing how earlier events led to later outcomes, but focus on the most recent status or decision if it supersedes earlier information.
</relevance_and_coherence>

<thinking_process>
When reorganizing the summary:
1. Have I cross-referenced all information to ensure all important unique details are captured? 
2. Have I identified and merged all duplicate information?
3. Have I created a chronological timeline of events, eliminating repetitions?
4. Have I ensured that the merged information remains accurate and context is preserved?
5. Does the reorganized summary present a coherent narrative of the entire document?
</thinking_process>

<coherence_check>
After completing the reorganized summary, review it to ensure:
1. The summary flows logically from one section to another.
2. There are no contradictions or inconsistencies in the presented information.
3. The overall narrative of the case is clear and easy to follow.
4. All sections contribute to a comprehensive understanding of the case.
5. Any relationships between different aspects of the case are clearly explained.
</coherence_check>

<warnings>
- Carefully review the entire concatenated summary before starting reorganization.
- Do not omit any unique information present in the original concatenated summary.
- When merging duplicate information, ensure all unique details from each instance are retained.
- If there are any contradictions in the duplicated information, include both versions and note the discrepancy.
- Maintain the original meaning and context of all information during reorganization.
</warnings>

<reference_materials>
## Concatenated Summary ##
{summaries}
</reference_materials>


<output_instructions>
If, for example, there are multiple allegations listed for different persons on the same case, group them under the same bullet point. 

If, for example, there are multiple references to the same procedural issue, group these mentions under one bullet point.

Your task is to return a condensed and coherent summary.

Present the condensed and combined summary in the following format and under the following headers: 

Legal Issues, Claims, Allegations, and Charges 
  • Bulletpoint 1.1 
  • Bulletpoint 1.2
  • Bulletpoint 1.3

Key Events, Incidents and Procedural Events
  • Bulletpoint 2.1
  • Bulletpoint 2.2
  • Bulletpoint 2.3

Main Findings, Decisions and Actions
  • Bulletpoint 3.1
  • Bulletpoint 3.2 
  • Bulletpoint 3.2 

Limit your response to these three headers. If an important point does not fit exactly in one of the headers, include it under the header that matches most closely. 

Provide the reorganized summary below. Ensure that all unique information from the concatenated summary is retained, duplicate information is consolidated, and the result is presented in a clear, logical manner. The length of the output may vary depending on the amount of unique information present - focus on completeness rather than brevity. After completing the summary, perform a final review using the coherence check guidelines to ensure the output presents a coherent and comprehensive description of the entire case.

</output_instructions>
"""


def format_content(content):
    # Remove extra whitespace and empty lines
    content = re.sub(r'\n\s*\n', '\n\n', content)
    content = re.sub(r' +', ' ', content)
    
    # Split content into lines
    lines = content.split('\n')
    
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line:
            # Check if the line is a header (e.g., all caps, or ends with a colon)
            if line.isupper() or line.endswith(':'):
                formatted_lines.append(f"\n{line}\n")
            else:
                formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def word_count(text):
    return len(text.split())

def load_and_split(file_path):
    logger.info(f"Processing document: {file_path}")

    with open(file_path, "r") as file:
        file_content = file.read()
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            data = {}

    logger.info(f"Keys in parsed JSON data: {data.keys()}")

    if "messages" in data and data["messages"]:
        docs = []
        original_page_number = 1
        for message in data["messages"]:
            page_content = message.get("page_content", "")
            if word_count(page_content) >= 50:
                formatted_content = format_content(page_content)
                doc = Doc(
                    page_content=formatted_content,
                    metadata={"seq_num": original_page_number},
                )
            else:
                doc = Doc(
                    page_content="No data to be processed on this page",
                    metadata={"seq_num": original_page_number},
                )
            docs.append(doc)
            original_page_number += 1

        logger.info(f"Data loaded and formatted from document: {file_path}")
        return docs
    else:
        logger.warning(f"No valid data found in document: {file_path}")
        return []


def create_memory_log(docs, pages_to_concatenate=2):
    memory_log = ""
    num_pages = len(docs)

    def concatenate_pages(start_idx, num_pages_to_concat):
        combined_content = ""
        current_idx = start_idx
        pages_concatenated = 0

        while pages_concatenated < num_pages_to_concat and current_idx < num_pages:
            page_content = docs[current_idx].page_content.replace("\n", " ")
            if "No data to be processed on this page" not in page_content:
                combined_content += page_content + " "
                pages_concatenated += 1
            current_idx += 1

        return combined_content.strip(), current_idx - start_idx

    def find_next_data_page(start_idx):
        current_idx = start_idx
        while current_idx < num_pages:
            content, pages_checked = concatenate_pages(current_idx, 1)
            if content:
                return current_idx
            current_idx += pages_checked
        return None

    # Process the first 10 pages with data
    start_idx = find_next_data_page(0)
    if start_idx is not None:
        pages_processed = 0
        while pages_processed < 5 and start_idx < num_pages:
            combined_content, pages_checked = concatenate_pages(start_idx, pages_to_concatenate)
            if combined_content:
                summary = process_memory_log_page(docs, start_idx, combined_content, 0, memory_log)["page_content"]
                memory_log = update_memory_log(memory_log, summary)
                print(f"Updated Memory Log: {memory_log}")
                pages_processed += 1
            start_idx += pages_checked

    # Process the last 10 pages with data
    end_idx = num_pages - 1
    pages_processed = 0
    while pages_processed < 5 and end_idx >= start_idx:
        last_data_idx = find_next_data_page(end_idx)
        if last_data_idx is None:
            break
        combined_content, pages_checked = concatenate_pages(last_data_idx, pages_to_concatenate)
        if combined_content:
            summary = process_memory_log_page(docs, last_data_idx, combined_content, 0, memory_log)["page_content"]
            memory_log = update_memory_log(memory_log, summary)
            pages_processed += 1
        end_idx = last_data_idx - 1

    return memory_log

def update_memory_log(memory_log, new_summary):
    memory_log_prompt = ChatPromptTemplate.from_template(memory_log_template)
    memory_log_chain = memory_log_prompt | llm | StrOutputParser()

    updated_memory_log = memory_log_chain.invoke(
        {"summary": new_summary, "memory_log": memory_log}
    )

    return updated_memory_log


def process_memory_log_page(docs, i, current_page, window_size, memory_log):
    prompt_response = ChatPromptTemplate.from_template(summary_template_for_memory_log)
    response_chain = prompt_response | llm | StrOutputParser()

    previous_page_ending = (
        docs[i - 1].page_content.replace("\n", " ")[-window_size:] if i > 0 else ""
    )
    next_page_beginning = (
        docs[i + 1].page_content.replace("\n", " ")[:window_size]
        if i < len(docs) - 1
        else ""
    )
    page_number = docs[i].metadata.get("seq_num")
    response = {"page_content": "", "page_number": page_number}
    if current_page:
        processed_content = response_chain.invoke(
            {
                "current_page": current_page,
            }
        )
        response["page_content"] = processed_content

    return response


def process_interval(interval_index, start_index, interval_size, docs, memory_log):
    end_index = start_index + interval_size
    interval_docs = docs[start_index:end_index]

    page_summaries = generate_summaries(interval_docs, memory_log)
    combined_summary, _ = process_summaries(page_summaries, memory_log)

    return combined_summary

def process_summaries(summaries, memory_log):
    combined_summary, updated_memory_log = combine_summaries(summaries, memory_log)
    return combined_summary, updated_memory_log


def process_file(filename, input_directory, output_directory, memory_log):
    json_path = os.path.join(input_directory, filename)
    docs = load_and_split(json_path)

    total_pages = len(docs)

    num_intervals = 4

    base_interval_size = total_pages // num_intervals
    extra_pages = total_pages % num_intervals

    interval_sizes = [
        base_interval_size + (1 if i < extra_pages else 0) for i in range(num_intervals)
    ]
    max_cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=max_cpus) as pool:
        process_interval_partial = partial(
            process_interval, docs=docs, memory_log=memory_log
        )
        interval_summaries = pool.starmap(
            process_interval_partial,
            [
                (i, sum(interval_sizes[:i]), interval_sizes[i])
                for i in range(num_intervals)
            ],
        )

    return interval_summaries

def process_batch(batch, num_pages_to_concat, memory_log):
    summary_prompt = ChatPromptTemplate.from_template(page_summary_template)
    verification_prompt = ChatPromptTemplate.from_template(page_summary_verification_template)
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    verification_chain = verification_prompt | llm | StrOutputParser()

    results = []
    i = 0
    batch_number = 1

    while i < len(batch):
        concat_pages = []
        page_numbers = []
        while len(concat_pages) < num_pages_to_concat and i < len(batch):
            current_page = batch[i].page_content.replace("\n", " ")
            if "No data to be processed on this page" not in current_page:
                concat_pages.append(current_page)
                page_numbers.append(batch[i].metadata.get("seq_num"))
            i += 1

        if concat_pages:
            original_document = " ".join(concat_pages)
        
            initial_summary = summary_chain.invoke({"current_page": original_document})
            
            # Verification step
            verified_summary = verification_chain.invoke({
                "original_document": original_document,
                "current_summary": initial_summary,
            })

            results.append(
                {"page_content": verified_summary, "page_numbers": page_numbers}
            )

        if not concat_pages:
            break

        batch_number += 1

    return results

def create_batches(docs, batch_size):
    return [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]


def generate_summaries(docs, memory_log, batch_size=20, num_pages_to_concat=10):
    batches = create_batches(docs, batch_size)

    results = []
    max_workers = os.cpu_count()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_batch, batch, num_pages_to_concat, memory_log)
            for batch in batches
        ]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    return results

def combine_summaries(summaries, memory_log):
    if not summaries:
        return None, memory_log

    combiner_llm = llm
    combiner_prompt = ChatPromptTemplate.from_template(combine_template)
    combiner_chain = combiner_prompt | combiner_llm | StrOutputParser()

    verification_llm = llm
    verification_prompt = ChatPromptTemplate.from_template(verification_template)
    verification_chain = verification_prompt | verification_llm | StrOutputParser()

    def combine_and_verify(summary_1, summary_2):
        combined_summary = combiner_chain.invoke({
            "summary_1": summary_1["page_content"],
            "summary_2": summary_2["page_content"]
        })
        verified_summary = verification_chain.invoke({
            "current_combined_summary": combined_summary,
            "summary_1": summary_1["page_content"],
            "summary_2": summary_2["page_content"]
        })
        return {
            "page_content": verified_summary,
            "page_numbers": summary_1["page_numbers"] + summary_2["page_numbers"]
        }

    def parallel_combine(summaries):
        if len(summaries) == 1:
            return summaries[0]
        elif len(summaries) == 2:
            return combine_and_verify(summaries[0], summaries[1])

        # Sort summaries based on the first page number in each summary
        sorted_summaries = sorted(summaries, key=lambda x: min(x["page_numbers"]))

        # Pair adjacent summaries
        pairs = [(sorted_summaries[i], sorted_summaries[i+1]) 
                 for i in range(0, len(sorted_summaries) - 1, 2)]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(combine_and_verify, *pair) for pair in pairs]
            combined_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # If there's an odd number of summaries, add the last one to the combined results
        if len(sorted_summaries) % 2 != 0:
            combined_results.append(sorted_summaries[-1])

        # Recursively combine the results
        return parallel_combine(combined_results)

    final_summary = parallel_combine(summaries)
    return final_summary, memory_log


def combine_and_verify(summary_1, summary_2, memory_log):
    combiner_llm = llm
    combine_prompt_template = ChatPromptTemplate.from_template(final_combine_template)
    combine_chain = combine_prompt_template | combiner_llm | StrOutputParser()

    verification_llm = llm
    verification_prompt_template = ChatPromptTemplate.from_template(
        final_verification_template
    )
    verification_chain = (
        verification_prompt_template | verification_llm | StrOutputParser()
    )

    improve_llm = llm
    improve_prompt_template = ChatPromptTemplate.from_template(improve_summary_template)
    improve_chain = improve_prompt_template | improve_llm | StrOutputParser()

    combined_summary = combine_chain.invoke(
        {"summary_1": summary_1, "summary_2": summary_2}
    )

    verified_summary = verification_chain.invoke(
        {
            "summary_1": summary_1,
            "summary_2": summary_2,
            "current_combined_summary": combined_summary,
        }
    )

    # improved summary
    improved_summary = improve_chain.invoke(
        {"summary": verified_summary, "memory_log": memory_log}
    )
    return improved_summary


def final_summarization(summaries, memory_log):
    if len(summaries) % 2 != 0:
        raise ValueError("The number of input summaries must be even.")
    
    if len(summaries) == 2:
        return [combine_and_verify(summaries[0], summaries[1], memory_log)]
    
    result = []
    mid = len(summaries) // 2
    
    first_half = combine_and_verify(summaries[0], summaries[1], memory_log)
    for i in range(2, mid):
        first_half = combine_and_verify(first_half, summaries[i], memory_log)
    result.append(first_half)
    
    second_half = combine_and_verify(summaries[mid], summaries[mid + 1], memory_log)
    for i in range(mid + 2, len(summaries)):
        second_half = combine_and_verify(second_half, summaries[i], memory_log)
    result.append(second_half)
    
    return result

def organize_final_summary(summaries):
    condense_llm = llm
    condense_prompt_template = ChatPromptTemplate.from_template(organization_template)
    condense_chain = condense_prompt_template | condense_llm | StrOutputParser()
    
    combined_summary = ""
    for i, summary in enumerate(summaries, 1):
        separator = f"############### SUMMARY {i} ###############\n"
        combined_summary += f"{separator}{summary}{separator}\n\n"
    
    summary = condense_chain.invoke({"summaries": combined_summary})
    return summary


def clean_summary(summary):
    # # Remove any introductory text followed by a colon
    cleaned_summary = re.sub(r'^[^:\n]+:\n', '', summary.strip())
    
    # Split the summary into lines
    summary_lines = cleaned_summary.split('\n')
    
    # Remove any empty lines at the beginning
    while summary_lines and not summary_lines[0].strip():
        summary_lines.pop(0)
    
    # Remove asterisks from each line
    summary_lines = [line.replace('*', '') for line in summary_lines]
    
    # Join the lines back together
    return '\n'.join(summary_lines).strip()


def save_summaries_to_json(summary, output_file, start_page, end_page):
    cleaned_summary = clean_summary(summary)
    output_data = [
        {
            "sentence": cleaned_summary,
            "filename": os.path.basename(output_file),
            "start_page": start_page,
            "end_page": end_page,
        }
    ]
    return {"files": output_data}

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Please provide the path to the directory, the selected model, and the output path as command-line arguments."
        )
        sys.exit(1)

    input_directory = sys.argv[1]
    selected_model = sys.argv[2]
    custom_template = sys.argv[3]
    output_path = sys.argv[4]
    output_data = []

    custom_template = str(custom_template)
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    try:
        for filename in os.listdir(input_directory):
            if filename.endswith(".json"):
                print(f"Processing file: {filename}")
                json_path = os.path.join(input_directory, filename)
                
                docs = load_and_split(json_path)
                memory_log = create_memory_log(docs)
                
                file_summaries = process_file(filename, input_directory, output_path, memory_log)
                final_summary = final_summarization(file_summaries, memory_log)
                condensed_summary = organize_final_summary(final_summary)
                
                start_page = docs[0].metadata["seq_num"]
                end_page = docs[-1].metadata["seq_num"]
                
                output_data.append(save_summaries_to_json(condensed_summary, filename, start_page, end_page))

        with open(output_path, "w") as output_file:
            json.dump(output_data, output_file, indent=4)

        end_time = time.time()
        total_time = end_time - start_time
        print("All files processed successfully.")
        print(f"Total execution time: {total_time:.2f} seconds")

    except Exception as e:
        print(f"Error processing JSON: {str(e)}")
        print(json.dumps({"success": False, "message": "Failed to process JSON"}))
        sys.exit(1)