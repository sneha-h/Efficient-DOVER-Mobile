# DOVER-VQA-Mini
Energy-Efficient DOVER Video Quality Assessment 

State-of-the-Art Model Comprehension
 
The Disentangled Objective Video Quality Evaluator (DOVER - https://github.com/VQAssessment/DOVER) is a cutting-edge model used for assessing the quality of User-Generated Content (UGC) videos. It evaluates videos based on two primary perspectives: technical and aesthetic. The technical perspective measures the perception of distortions in the video, while the aesthetic perspective focuses on user preferences and content recommendations. The DOVER model has shown significant potential in delivering high-quality assessments of UGC videos by balancing these two perspectives.
 
The primary challenge addressed in this project is the reduction of the DOVER model's size and runtime without compromising its quality evaluation efficiency. This is crucial for enhancing the model's usability in real-world applications and to enable practical and widespread adoption for efficient and effective quality assessment of UGC-VQA videos. 

 
Pruning was implemented, reducing 50% of the parameters in the DOVER model. The pre-pruning model had a size of 214.46 MB with 56,218,302 parameters. Post-pruning, the model size was reduced to 108.58 MB with 28,462,574 parameters, achieving a sparsity of approximately 49.37%.
 
   - Validation accuracies before and after pruning were compared:
     Pre-Pruning 
     - DOVER Model-S:
       - SROCC: 0.8997
       - PLCC: 0.9042
     - DOVER Model-N:
       - SROCC: 0.9044
       - PLCC: 0.9094
    Post-Pruning
     - DOVER Model-S:
       - SROCC: 0.9022
       - PLCC: 0.9137
     - DOVER Model-N:
       - SROCC: 0.8971
       - PLCC: 0.9092

Future Work

Further experiments will be conducted to increase the pruning percentage and test the effectiveness on both DOVER-Mobile and DOVER models. The goal is to continue optimizing the balance between model size, runtime efficiency, and quality assessment accuracy
