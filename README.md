# DOVER-VQA-Mini
Energy-Efficient DOVER Video Quality Assessment 

# Introduction
State-of-the-Art Model Comprehension
 
The Disentangled Objective Video Quality Evaluator (DOVER - https://github.com/VQAssessment/DOVER) is a cutting-edge model used for assessing the quality of User-Generated Content (UGC) videos. It evaluates videos based on two primary perspectives: technical and aesthetic. The technical perspective measures the perception of distortions in the video, while the aesthetic perspective focuses on user preferences and content recommendations. The DOVER model has shown significant potential in delivering high-quality assessments of UGC videos by balancing these two perspectives.
 
The primary challenge addressed in this project is the reduction of the DOVER model's size and runtime without compromising its quality evaluation efficiency. This is crucial for enhancing the model's usability in real-world applications and to enable practical and widespread adoption for efficient and effective quality assessment of UGC-VQA videos. 

There is a growing interest in pruning techniques for various types of neural networks within the research community. However, the application of these techniques to Visual Question Answering (VQA) models remains relatively underexplored and fragmented.

# Proposed pruning technique:
This project introduces and evaluates a novel layer-specific pruning method tailored to VQA models. By combining this approach with channel pruning, we achieve a significant reduction in model size while maintaining, or even slightly improving, performance.
The experiments are conducted on state of the art VQA model - DOVER-Mobile 
Model:
https://arxiv.org/abs/2211.04894
https://github.com/VQAssessment/DOVER

The proposed method reduces DOVER-Mobile model parameters by approximately 79.7%, FLOPs by 79.7%, latency by 29.9%, and memory usage by 64.9%, all while slightly improving accuracy and the score.

 ![Screenshot 2024-08-28 at 01 33 37](https://github.com/user-attachments/assets/253d7e48-c123-44e9-9fc4-ca814067afb4)

# Datasets:
The download links for the videos are as follows:

📖 KoNViD-1k: Official Site<br />
📖 LIVE-VQC: Official Site


# Install:
git clone https://github.com/QualityAssessment/Efficiant-DOVER-Mobile.git<br />
cd Efficient-DOVER/DOVER<br />
pip install -e .<br />
mkdir pretrained_weights<br />
cd pretrained_weights<br />
wget https://github.com/QualityAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth<br />
cd ..

# Training: Pruned DOVER-Mobile models
cp all python and yml files to DOVER<br />
cd DOVER<br />
Structured pruning:<br />
  python transfer_learning_structured.py -t <dataset_name> -o dover-mobile.yml<br />
Unstructured pruning:<br />
  python transfer_learning_unstructured.py -t <dataset_name> -o dover-mobile.yml<br />
Channel pruning:<br />
  python transfer_learning_channel.py -t <dataset_name> -o dover-mobile.yml<br />
Layer specific pruning:<br />
  python transfer_learning_layer_specific.py -t <dataset_name> -o dover-mobile.yml<br />
Layer specific and channel pruning:<br />
  python transfer_learning_layer_specific_with_channel.py -t <dataset_name> -o dover-mobile.yml<br />

Eg:<br />
python  transfer_learning_structured.py -t val-kv1k for Structured pruning with KoNViD-1k dataset.<br />
python  transfer_learning_structured.py -t val-livevqc for for Structured pruning with LIVE-VQC dataset.<br />

As the backbone will not be updated here, the checkpoint saving process will only save the regression heads with smaller size compared to full model. To use it, simply replace the head weights.
For end-to-end fine-tune right now (by modifying the num_epochs: 0 to num_epochs: 15 in ./dover-mobile.yml). It will require more memory cost and more storage cost for the weights (with full parameters) saved, but will result in optimal accuracy.


# Future Work

Future work could focus on applying the developed pruning techniques to other state-of-the-art VQA models to validate their effectiveness further. Expanding these methods beyond the DOVER-Mobile model would help assess their generalizability across different architectures. Additionally, exploring the combination of pruning strategies with other model compression techniques like quantization and knowledge distillation could lead to even more efficient models. Optimizing the balance between different pruning approaches and testing these pruned models in real-world settings, such as mobile devices, could provide valuable insights for improving performance and usability in practical applications.
