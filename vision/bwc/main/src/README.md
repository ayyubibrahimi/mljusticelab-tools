# Flow

Goal: For a given UOF case, which includes a report written by an officer and body-worn-camera footage, we want to compare the report written by the officer with the body worn camera footage to identify potential discrepancies. 

Stage 1: 
- Read the UOF report. 
- Determine the incident type. 
- For example, a typical UOF report might say something like "After person X refused to listen to my order to stop walking toward me, I pulled out my pepperspray used it on person X."

Stage 2:
- Transcribe the BWC. 
- Based on the audio transcription, chose a time interval that likely corresponds with the beginning of the incident. 

Stage 3: 
- For the chosen time interval, review the video for to help determine if the interval is correct. 
- For example, use a binary classifier to determine if we can see someone being peppersprayed as they walk toward the camera. 
- If the interval seems correct, we create a summary of the incident based on both the audio and video. This summary should then be compared with the written report to identify potential discrepancies. For example, the written report may say that the officer gave person X 5 warnings before pulling out the pepperspray, while the AV might show that the officer gave 1 warning after the pepperspray was already pulled. 
- If the time interval is not correct, step to a different time interval and/or increase the number of frames per second 

Optimization qs:
Thought 1:
- We want to find an optimal timestep and number of frames per second based based on what's returned by the first iteration.

Thought 2:
- We need to decide in which direction we search (inputs are start_time, end_time). 
- Currently, we can receive signal about which direction to step depending on whether or not a given frame contains data of interest. 
For example, if the image contains a taser, True will be returned. The next iteration will step toward the time interval where we see True values. How much we step in this direction depends on the density of True values. If there are no True values, we step in both directions. 

Thought 3:
- We need to determine how much much to increase the time interval param by (10s more, 20s more, 30s, etc). Make this dynamic.

Thought 4:L
- We need to determine how much to adjust the frames per second param (can we only adjust in area around true positives after confirming that uping fps does not improve performance in areas where no true values are returned)
