# 

**1.** Add network analysis 
**2.** Add option to choose different models 
**3.** Add option to extract victim data
**4.** Add option to process long documents 
**5.** Add option to classify docs 
**6.** Add option to adjust prompt
**7.** explore the 10-15 page window. Can we go larger or smaller? And does it work for osm

**Quantitative metrics**
- BLEU, the simplest metric, calculates the degree of overlap between the reference and generated texts by considering 1- to 4-gram sequences;
- ROUGE-L evaluates similarity based on the longest common subsequence; 
- BERTScore, which leverages contextual BERT embeddings to evaluate the semantic similarity of the generated and reference texts;

- Completeness: “Which summary more completely captures important information?” This compares the summaries’ recall, i.e. the amount of clinically significant detail retained from the input text.
- Correctness: “Which summary includes less false information?” This compares the summaries’ precision, i.e. instances of fabricated information.
- Conciseness: “Which summary contains less non-important information?” This compares which summary is more condensed, as the value of a summary decreases with superfluous information.


