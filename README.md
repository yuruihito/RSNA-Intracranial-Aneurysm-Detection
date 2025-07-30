🧠 RSNA 2023 Brain Aneurysm Detection Challenge
🚀 Overview | 概要
English
Welcome to the RSNA 2023 Brain Aneurysm Detection Challenge! This competition is a vital initiative to revolutionize the early detection of intracranial aneurysms—a silent and potentially deadly condition. Affecting approximately 3% of the global population, these aneurysms often go unnoticed until they rupture, leading to severe illness or death. Annually, around 500,000 lives are lost worldwide due to ruptured aneurysms, with nearly half of the victims being under 50 years old.

Our mission, in collaboration with the American Society of Neuroradiology (ASNR), the Society of Neurointerventional Surgery (SNIS), and the European Society of Neuroradiology (ESNR), is to develop advanced Machine Learning models. These models will accurately detect and precisely localize intracranial aneurysms across various medical imaging modalities, including CTA, MRA, T1 post-contrast, and T2-weighted MRI. This challenge emphasizes real-world clinical variability, incorporating data from diverse institutions, scanners, and imaging protocols to test the generalizability of your models.

Your contributions will be instrumental in paving the way for automated, accurate, and efficient diagnostic solutions. Ultimately, this will enable earlier interventions, saving countless lives by preventing catastrophic aneurysm ruptures.

日本語訳を表示
🎯 Clinical Problem & Goal | 臨床上の問題と目標
English
Intracranial aneurysms are localized abnormal dilations of brain arteries. While often asymptomatic until rupture, even small aneurysms pose a significant risk, potentially leading to subarachnoid hemorrhage (SAH)—a severe type of stroke. SAH is the most common cause of non-traumatic SAH and accounts for 3% of all strokes and 5% of stroke deaths. Early detection is crucial, as minimally-invasive treatments can often be life-saving.

This challenge primarily focuses on detecting saccular aneurysms, the most common form, and aims for both their detection and precise localization across the entire brain. Your models will need to identify the presence or absence of aneurysms within 13 specific anatomical locations for each imaging series.

日本語訳を表示
📸 Imaging Modalities | 画像診断モダリティ
English
We're leveraging a diverse set of medical imaging modalities to provide a comprehensive view of the brain vasculature. This includes:

Computed Tomography Angiography (CTA): A non-invasive technique that visualizes blood vessels and surrounding tissues. While faster and less risky than DSA, it involves ionizing radiation and has lower spatial resolution.

Magnetic Resonance Angiography (MRA): A valuable alternative that avoids ionizing radiation and iodinated contrast. While generally safer, it has lower spatial resolution and longer scan times.

T1 Post-Contrast MRI & T2-Weighted MRI: Though not typically used for aneurysm evaluation, these commonly acquired sequences are included to explore opportunistic screening possibilities.

日本語訳を表示
📊 Evaluation Metric | 評価指標
English
Submissions are evaluated based on a weighted multilabel Area Under the ROC Curve (AUC ROC). For each of the fourteen target labels, an AUC ROC score is computed. The score for "Aneurysm Present" is weighted by 13, while all other 13 location-specific scores are weighted by 1. The final score is the average of these fourteen weighted AUC ROC scores.

Mathematically, the final score is represented as:

Final Score= 
2
AUC 
Aneurysm Present
​
 +average(AUC 
other 13 scores
​
 )
​
 
You can find the metric code here.

日本語訳を表示
📁 Dataset Details | データセットの詳細
English
The dataset is rich, containing not only imaging data (DICOM images with segmentation labels for a subset) but also two crucial CSV files: train.csv and train_localizers.csv.

train.csv: Contains primary training labels for each imaging series, including:

SeriesInstanceUID, PatientAge, PatientSex, Modality

13 Location-Specific Aneurysm Labels: Binary (0/1) indicating presence/absence in specific anatomical sites.

Aneurysm Present: Overall binary (0/1) indicating if any aneurysm is present in the series.

train_localizers.csv: Provides precise localization data for individual aneurysms in the training set, linking SeriesInstanceUID and SOPInstanceUID with coordinates (x, y) and a location description.

日本語訳を表示
✨ Why Participate? | 参加する意義
English
Save Lives: Your model could directly contribute to earlier diagnosis and intervention, preventing devastating aneurysm ruptures.

Real-World Impact: Work with diverse, clinically varied data that mimics real hospital settings, enhancing your model's robustness and generalizability.

Learn & Grow: Explore cutting-edge medical imaging analysis, machine learning techniques, and collaborate with a global community of experts.

Let's make a difference together! Join the challenge and help us build a healthier future.

日本語訳を表示
🤝 How to Contribute | 貢献方法
English
Clone the Repository:

Bash

git clone https://github.com/your-username/RSNA-Aneurysm-Detection.git
cd RSNA-Aneurysm-Detection
Set Up Your Environment: (Recommended: Python 3.9+)

Bash

pip install -r requirements.txt
Explore the Data: Dive into the data/ directory and the provided train.csv and train_localizers.csv.

Develop Your Model: Create your ML models for aneurysm detection and localization.

Submit Your Solution: Utilize the provided evaluation API. Refer to the example notebook for detailed submission instructions.

Share Your Insights: Feel free to open issues or pull requests to share your findings and improvements!

日本語訳を表示
