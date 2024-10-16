# Flow

Stage 1: 
- Read the report. Determine the incident type, for example, an incident where pepperspray was used

Stage 2:
- Transcribe the BWC. 
- Based on the audio transcription, chose a time interval that likely corresponds with the beginning of the use of force

Stage 3: 
- Review the video for this time interval to determine if that is captured by our interval.
- For example, for each frame, use a binary classifier to determine if we can see someone being peppersprayed 
- If the interval is correct, we create a summary of the incident based on the audio, video, and compare it to the report 
- If not, step to a different time interval 

Notes:
- We want to find the time step based on the distribution from the first iteration (where are the TRUE values)
- Based on this distribution, we need to decide in which direction we search
- Based on this distribution, we need to determine how much much we search (10s more, 20s more, 30s, etc) temporal space we search
- Based on this distrubition, we need to be able to determine how to adjust how many fps we choose
