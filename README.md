# 🧠 RSNA Brain Aneurysm Detection Challenge

---

## 🚀 Overview | 概要

### English
This project addresses the critical challenge of accurate and timely detection of intracranial aneurysms. Often silent until rupture, these localized arterial dilations in the brain pose a significant, life-threatening risk. A ruptured aneurysm can lead to subarachnoid hemorrhage, a severe form of stroke with high morbidity and mortality rates. Our goal is to develop advanced machine learning models that can detect and precisely localize these aneurysms across various medical imaging modalities, ultimately contributing to earlier intervention and improved patient outcomes.

<details>
<summary>日本語訳を表示</summary>

このプロジェクトは、頭蓋内動脈瘤の正確かつタイムリーな検出という極めて重要な課題に取り組んでいます。脳内の動脈の局所的な拡張である動脈瘤は、破裂するまで無症状であることが多く、生命を脅かす重大なリスクを伴います。動脈瘤が破裂すると、くも膜下出血という重篤な脳卒中を引き起こし、高い罹患率と死亡率を伴います。私たちの目標は、様々な医用画像モダリティにおいてこれらの動脈瘤を検出し、その位置を正確に特定できる高度な機械学習モデルを開発し、最終的に早期介入と患者の転帰改善に貢献することです。
</details>

---

## 🎯 Clinical Problem & Goal | 臨床上の問題と目標

### Prevalence and Risk:

Prevalence and Risk:

They affect an estimated 3% of the global population, with 15-30% of affected individuals having multiple aneurysms.

Alarmingly, up to 50% are only diagnosed after they rupture. This event, known as subarachnoid hemorrhage (SAH), is a life-threatening type of stroke caused by bleeding into the subarachnoid space.

A ruptured aneurysm is the most common cause of non-traumatic SAH, accounting for 3% of all strokes and 5% of stroke deaths.

Worldwide, ruptured aneurysms cause approximately 500,000 deaths annually, with nearly half of the victims being under 50.

Symptoms and Challenges in Detection:

Most intracranial aneurysms are asymptomatic until they rupture. The rupture often presents as a severe "thunderclap" headache, reduced consciousness, and can be fatal if untreated.

Larger aneurysms may occasionally cause symptoms before rupture by pressing on adjacent nerves.

Detection is challenging because aneurysms are often small and remain asymptomatic. However, even small aneurysms carry a rupture risk.

Treatment and the Role of Early Detection:

When detected, aneurysms can often be treated with minimally-invasive procedures like endovascular coiling or surgical clipping, which can be life-saving.

While the management of small, asymptomatic aneurysms remains controversial, early detection allows for careful monitoring and timely intervention, significantly reducing the risk of catastrophic rupture.

This project specifically targets the identification of saccular aneurysms, focusing on both their detection and precise localization anywhere within the brain. Other aneurysm types (e.g., fusiform, pseudoaneurysms) are excluded due to their distinct imaging appearances and risk profiles.

<details>
<summary>日本語訳を表示</summary>

脳動脈瘤（脳内動脈の局所的な異常拡張）は、臨床上極めて重要な問題です。最も一般的なのは**嚢状動脈瘤（または「ベリー動脈瘤」）**で、通常は動脈の分岐部に発生する、丸みを帯びた分葉状の突出として現れます。

罹患率とリスク:

**世界人口の推定3%**が罹患しており、これらの患者の15～30%では複数の動脈瘤が見つかります。

驚くべきことに、最大50%が破裂後に初めて診断されます。この事象は**くも膜下出血（SAH）**として知られ、くも膜下腔への出血によって引き起こされる生命を脅かすタイプの脳卒中です。

動脈瘤破裂は非外傷性くも膜下出血の最も一般的な原因であり、全脳卒中の3%と脳卒中による死亡の5%を占めます。

世界中で、動脈瘤破裂により年間約50万人が死亡しており、犠牲者の約半数は50歳未満です。

症状と検出の課題:

ほとんどの脳動脈瘤は破裂するまで無症状です。破裂はしばしば激しい「雷鳴頭痛」、意識障害として現れ、治療せずに放置すれば致命的になる可能性があります。

より大きな動脈瘤は、隣接する神経を圧迫するなどして、破裂前に症状を引き起こすことがあります。

検出が困難なのは、動脈瘤が小さく、無症状であることが多いからです。しかし、小さな動脈瘤であっても破裂のリスクがあります。

治療と早期発見の役割:

検出された場合、動脈瘤は血管内コイル塞栓術や外科的クリッピングのような低侵襲の手術で治療できることが多く、これらは命を救う可能性があります。

小さく無症状の動脈瘤の管理については議論の余地がありますが、早期に発見することで注意深い経過観察とタイムリーな介入が可能になり、壊滅的な破裂のリスクを大幅に低減できます。

このプロジェクトは、嚢状動脈瘤の特定に特化しており、脳内のどこにでも存在する動脈瘤の検出と正確な局在化の両方を目標としています。他のタイプの動脈瘤（紡錘状動脈瘤、仮性動脈瘤など）は、その画像上の見え方やリスクプロファイルが異なるため、対象外とします。

</details>

---

## 🧠 Anatomical Context: Mapping the Aneurysms | 解剖学的背景：動脈瘤の部位特定

### English
Understanding the brain's arterial supply is crucial for accurate aneurysm localization. The brain receives blood from two main circulations:

Anterior Circulation: Primarily supplied by the Internal Carotid Arteries (ICAs). For this challenge, the ICA is segmented into supraclinoid and infraclinoid portions due to the clinical significance of the dura mater entry point. Major branches include the Middle Cerebral Arteries (MCAs) and Anterior Cerebral Arteries (ACAs).

Posterior Circulation: Supplied by the Vertebral Arteries (VAs), which merge to form the Basilar Artery (BA). Key branches are the Posterior Inferior, Anterior Inferior, Superior Cerebellar Arteries, and Posterior Cerebral Arteries. For our task, the posterior circulation is divided into the Basilar Tip and the rest of the posterior circulation.

These circulations communicate at the base of the brain through the Circle of Willis, a critical arterial connection providing collateral blood flow. This circle involves parts of the bilateral ACAs and PCAs, linked by the Anterior Communicating Artery (ACom) and paired Posterior Communicating Arteries (PComs).

Target Locations for Prediction:
Our models aim to predict the presence or absence of aneurysms in 13 specific anatomical locations for each imaging series. These locations are critical for clinical diagnosis and treatment planning:

Left Infraclinoid Internal Carotid Artery

Right Infraclinoid Internal Carotid Artery

Left Supraclinoid Internal Carotid Artery

Right Supraclinoid Internal Carotid Artery

Left Middle Cerebral Artery

Right Middle Cerebral Artery

Anterior Communicating Artery

Left Anterior Cerebral Artery

Right Anterior Cerebral Artery

Left Posterior Communicating Artery

Right Posterior Communicating Artery

Basilar Tip

Other Posterior Circulation (e.g., mid-basilar, vertebral, PICA, AICA, SCA, PCA)

<details>
<summary>日本語訳を表示</summary>

脳動脈瘤の正確な局在化には、脳の動脈供給を理解することが不可欠です。脳は主に2つの循環から血液供給を受けています。

前部循環： 主に**内頚動脈（ICA）**によって供給されます。このチャレンジでは、硬膜への進入点の臨床的意義から、ICAは硬膜上部分と硬膜下部分に分けられます。主要な枝には、中大脳動脈（MCA）と前大脳動脈（ACA）が含まれます。

後部循環： 椎骨動脈（VA）から供給され、椎骨動脈は合流して脳底動脈（BA）を形成します。主要な枝には、後下小脳動脈、前下小脳動脈、上小脳動脈、後大脳動脈があります。私たちの課題では、後部循環は脳底動脈先端と残りの後部循環に分けられます。

これらの循環は、脳底にあるウィリス動脈輪を介して連絡しており、側副血行路を維持する上で重要な動脈の連結です。この動脈輪は、前交通動脈（ACom）と対をなす後交通動脈（PCom）によって連結された左右のACAとPCAの一部で構成されます。

予測対象の解剖学的場所:
私たちのモデルは、各画像シリーズについて、13の特定の解剖学的場所における動脈瘤の有無を予測することを目指しています。これらの場所は、臨床診断と治療計画にとって重要です。

左硬膜下内頚動脈

右硬膜下内頚動脈

左硬膜上内頚動脈

右硬膜上内頚動脈

左中大脳動脈

右中大脳動脈

前交通動脈

左前大脳動脈

右前大脳動脈

左後交通動脈

右後交通動脈

脳底動脈先端

その他の後部循環（例：脳底動脈中央部、椎骨動脈、PICA、AICA、SCA、PCA）

</details>

---

## 📸 Imaging Modalities for Aneurysm Detection | 動脈瘤検出のための画像診断モダリティ

### English
Aneurysms can be identified using various imaging modalities, each with its own advantages and limitations:

Digital Subtraction Angiography (DSA): Generally considered the "gold standard" due to its high spatial and temporal resolution, especially with 3D acquisitions. However, it's invasive, requiring catheter insertion.

Computed Tomography Angiography (CTA): A non-invasive and faster alternative to DSA. It provides good visualization of blood vessels and surrounding brain tissue. However, it uses ionizing radiation and iodinated contrast, and has lower spatial resolution compared to DSA, making small aneurysms harder to detect. It images at a single time point, limiting flow evaluation.

Example: Middle cerebral artery aneurysm visible on CTA

Magnetic Resonance Angiography (MRA): A valuable non-invasive option that avoids ionizing radiation and iodinated contrast. It can be performed with or without contrast agents. Limitations include lower spatial resolution, longer scan times, and contraindications for patients with certain implants.

Example: Middle cerebral artery aneurysm visible on MRA

T1 Post-Contrast MRI & T2-Weighted MRI: While not typically used as primary aneurysm evaluation sequences, aneurysms can still be visible on them. Including these commonly acquired MRI sequences in the dataset offers an opportunity for opportunistic screening from routine brain imaging studies.

Example: Anterior communicating artery aneurysm visible on T1 post-contrast and T2-weighted MRI

This project leverages data from CTA, MRA, and T1 post-contrast/T2-weighted MRI to develop robust models capable of detecting saccular aneurysms across diverse clinical imaging settings.

<details>
<summary>日本語訳を表示</summary>

提出されたモデルは、**重み付けされた多ラベルROC曲線下面積（AUC ROC）**によって評価されます。14のターゲットラベルそれぞれについてAUC ROCスコアが計算され、「**Aneurysm Present（動脈瘤の有無）**」のスコアには**13の重み**が割り当てられ、他の13の場所固有のスコアすべてには1の重みが割り当てられます。最終スコアは、これらの重み付けされた14のAUC ROCスコアの平均です。

数学的には、最終スコアは次のように表されます。

$$\text{最終スコア} = \frac{\text{AUC}_{\text{動脈瘤の有無}} + \text{平均}(\text{AUC}_{\text{他の13スコア}})}{2}$$

評価指標のコードは[こちら](https://www.kaggle.com/code/awsaf49/mean-weighted-columnwise-aucroc)で確認できます。

</details>

---

## 📁 Dataset Details | データセットの詳細

### English
The dataset is rich, containing not only imaging data (DICOM images with segmentation labels for a subset) but also two crucial CSV files: `train.csv` and `train_localizers.csv`.

* **`train.csv`**: Contains primary training labels for each imaging series, including:
    * `SeriesInstanceUID`, `PatientAge`, `PatientSex`, `Modality`
    * **13 Location-Specific Aneurysm Labels**: Binary (0/1) indicating presence/absence in specific anatomical sites.
    * `Aneurysm Present`: Overall binary (0/1) indicating if any aneurysm is present in the series.

* **`train_localizers.csv`**: Provides precise localization data for individual aneurysms in the training set, linking `SeriesInstanceUID` and `SOPInstanceUID` with `coordinates (x, y)` and a `location` description.

<details>
<summary>日本語訳を表示</summary>

このデータセットは豊富で、画像データ（DICOM画像と一部のケースにはセグメンテーションラベル）だけでなく、`train.csv`と`train_localizers.csv`という2つの重要なCSVファイルが含まれています。

* **`train.csv`**: 各画像シリーズの主要なトレーニングラベルが含まれます。
    * `SeriesInstanceUID`、`PatientAge`、`PatientSex`、`Modality`
    * **13の場所固有の動脈瘤ラベル**：特定の解剖学的部位における動脈瘤の有無を示すバイナリ（0/1）ラベル。
    * `Aneurysm Present`：シリーズ内に動脈瘤が全く存在するかどうかを示す全体のバイナリ（0/1）ラベル。

* **`train_localizers.csv`**: トレーニングセット内の個々の動脈瘤の正確な局在化データを提供し、`SeriesInstanceUID`および`SOPInstanceUID`を`coordinates (x, y)`と`location`記述にリンクさせます。

</details>

---
