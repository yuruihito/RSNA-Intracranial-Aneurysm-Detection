# 🧠 RSNA 2023 Brain Aneurysm Detection Challenge

---

## 🚀 Overview | 概要

### English
Welcome to the RSNA 2023 Brain Aneurysm Detection Challenge! This competition is a vital initiative to **revolutionize the early detection of intracranial aneurysms**—a silent and potentially deadly condition. Affecting approximately 3% of the global population, these aneurysms often go unnoticed until they rupture, leading to severe illness or death. Annually, around 500,000 lives are lost worldwide due to ruptured aneurysms, with nearly half of the victims being under 50 years old.

Our mission, in collaboration with the American Society of Neuroradiology (ASNR), the Society of Neurointerventional Surgery (SNIS), and the European Society of Neuroradiology (ESNR), is to develop **advanced Machine Learning models**. These models will accurately detect and precisely localize intracranial aneurysms across various medical imaging modalities, including CTA, MRA, T1 post-contrast, and T2-weighted MRI. This challenge emphasizes **real-world clinical variability**, incorporating data from diverse institutions, scanners, and imaging protocols to test the generalizability of your models.

Your contributions will be instrumental in paving the way for automated, accurate, and efficient diagnostic solutions. Ultimately, this will enable earlier interventions, **saving countless lives** by preventing catastrophic aneurysm ruptures.

<details>
<summary>日本語訳を表示</summary>

RSNA 2023 脳動脈瘤検出チャレンジへようこそ！このコンペティションは、**脳動脈瘤の早期発見を革新する**ための重要な取り組みです。脳動脈瘤は、世界人口の約3%に影響を及ぼす、初期には無症状で潜在的に命に関わる病態です。破裂するまで発見されないことが多く、破裂すると重篤な病気や死に至ります。毎年、世界中で約50万人が動脈瘤破裂で命を落としており、そのほぼ半数が50歳未満です。

本コンペティションは、米国神経放射線学会（ASNR）、神経血管内治療学会（SNIS）、欧州神経放射線学会（ESNR）との共同開催で、**高度な機械学習モデルの開発**を目指します。これらのモデルは、CTA、MRA、T1造影後、T2強調MRIなど、様々な医用画像モダリティにおいて、脳動脈瘤を正確に検出し、その位置を特定します。このチャレンジでは、**実世界の臨床的変動**が重視されており、多様な医療機関、スキャナー、画像プロトコルからのデータが組み込まれ、モデルの汎用性が試されます。

皆さんの貢献は、自動化された正確かつ効率的な診断ソリューションへの道を切り開く上で不可欠です。最終的には、早期介入を可能にすることで、**壊滅的な動脈瘤破裂を防ぎ、数え切れない命を救う**ことにつながるでしょう。

</details>

---

## 🎯 Clinical Problem & Goal | 臨床上の問題と目標

### English
**Intracranial aneurysms** are localized abnormal dilations of brain arteries. While often asymptomatic until rupture, even small aneurysms pose a significant risk, potentially leading to **subarachnoid hemorrhage (SAH)**—a severe type of stroke. SAH is the most common cause of non-traumatic SAH and accounts for 3% of all strokes and 5% of stroke deaths. Early detection is crucial, as minimally-invasive treatments can often be life-saving.

This challenge primarily focuses on detecting **saccular aneurysms**, the most common form, and aims for both their **detection and precise localization** across the entire brain. Your models will need to identify the presence or absence of aneurysms within **13 specific anatomical locations** for each imaging series.

<details>
<summary>日本語訳を表示</summary>

**脳動脈瘤**は、脳動脈の局所的な異常な拡張です。破裂するまで無症状であることが多いですが、小さな動脈瘤であっても、**くも膜下出血（SAH）**という重篤な脳卒中を引き起こす可能性があります。くも膜下出血は非外傷性SAHの最も一般的な原因であり、全脳卒中の3%、脳卒中による死亡の5%を占めます。低侵襲治療が命を救うことが多いため、早期発見が極めて重要です。

このチャレンジは、最も一般的な形態である**嚢状動脈瘤の検出**に主に焦点を当てており、脳全体におけるその**検出と正確な局在化**の両方を目指しています。あなたのモデルは、各画像シリーズについて、**13の特定の解剖学的場所**における動脈瘤の有無を識別する必要があります。

</details>

---

## 📸 Imaging Modalities | 画像診断モダリティ

### English
We're leveraging a diverse set of medical imaging modalities to provide a comprehensive view of the brain vasculature. This includes:

* **Computed Tomography Angiography (CTA):** A non-invasive technique that visualizes blood vessels and surrounding tissues. While faster and less risky than DSA, it involves ionizing radiation and has lower spatial resolution.
* **Magnetic Resonance Angiography (MRA):** A valuable alternative that avoids ionizing radiation and iodinated contrast. While generally safer, it has lower spatial resolution and longer scan times.
* **T1 Post-Contrast MRI & T2-Weighted MRI:** Though not typically used for aneurysm evaluation, these commonly acquired sequences are included to explore opportunistic screening possibilities.

<details>
<summary>日本語訳を表示</summary>

脳血管系を包括的に把握するため、様々な医用画像モダリティを活用します。これには以下が含まれます：

* **CT血管造影（CTA）：** 血管と周囲組織を視覚化する非侵襲的手法です。DSAよりも高速でリスクが低いですが、電離放射線を使用し、空間分解能が低いです。
* **磁気共鳴血管造影（MRA）：** 電離放射線とヨード造影剤の使用を避ける貴重な代替手段です。一般的に安全ですが、空間分解能が低く、スキャン時間が長いです。
* **T1造影後MRI & T2強調MRI：** 通常、動脈瘤の評価には使用されませんが、これらの一般的なシーケンスは、機会的スクリーニングの可能性を探るために含まれています。

</details>

---

## 📊 Evaluation Metric | 評価指標

### English
Submissions are evaluated based on a **weighted multilabel Area Under the ROC Curve (AUC ROC)**. For each of the fourteen target labels, an AUC ROC score is computed. The score for **"Aneurysm Present" is weighted by 13**, while all other 13 location-specific scores are weighted by 1. The final score is the average of these fourteen weighted AUC ROC scores.

Mathematically, the final score is represented as:

$$\text{Final Score} = \frac{\text{AUC}_{\text{Aneurysm Present}} + \text{average}(\text{AUC}_{\text{other 13 scores}})}{2}$$

You can find the metric code [here](https://www.kaggle.com/code/awsaf49/mean-weighted-columnwise-aucroc).

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

## ✨ Why Participate? | 参加する意義

### English
* **Save Lives:** Your model could directly contribute to earlier diagnosis and intervention, preventing devastating aneurysm ruptures.
* **Real-World Impact:** Work with diverse, clinically varied data that mimics real hospital settings, enhancing your model's robustness and generalizability.
* **Learn & Grow:** Explore cutting-edge medical imaging analysis, machine learning techniques, and collaborate with a global community of experts.

Let's make a difference together! Join the challenge and help us build a healthier future.

<details>
<summary>日本語訳を表示</summary>

* **命を救う：** あなたのモデルが早期診断と介入に直接貢献し、壊滅的な動脈瘤破裂を防ぐ可能性があります。
* **実世界への影響：** 実際の病院環境を模倣した多様な臨床データに取り組み、モデルの堅牢性と汎用性を高めます。
* **学びと成長：** 最先端の医用画像解析、機械学習技術を探求し、世界中の専門家コミュニティと協力できます。

一緒に変化を生み出しましょう！チャレンジに参加して、より健康な未来を築く手助けをしてください。

</details>

---

## 🤝 How to Contribute | 貢献方法

### English
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/RSNA-Aneurysm-Detection.git](https://github.com/your-username/RSNA-Aneurysm-Detection.git)
    cd RSNA-Aneurysm-Detection
    ```
2.  **Set Up Your Environment:** (Recommended: Python 3.9+)
    ```bash
    pip install -r requirements.txt
    ```
3.  **Explore the Data:** Dive into the `data/` directory and the provided `train.csv` and `train_localizers.csv`.
4.  **Develop Your Model:** Create your ML models for aneurysm detection and localization.
5.  **Submit Your Solution:** Utilize the provided evaluation API. Refer to the [example notebook](https://www.kaggle.com/code/awsaf49/rsna-2023-submission-example) for detailed submission instructions.
6.  **Share Your Insights:** Feel free to open issues or pull requests to share your findings and improvements!

<details>
<summary>日本語訳を表示</summary>

1.  **リポジトリをクローンする：**
    ```bash
    git clone [https://github.com/your-username/RSNA-Aneurysm-Detection.git](https://github.com/your-username/RSNA-Aneurysm-Detection.git)
    cd RSNA-Aneurysm-Detection
    ```
2.  **環境をセットアップする：** (推奨：Python 3.9+)
    ```bash
    pip install -r requirements.txt
    ```
3.  **データを探索する：** `data/`ディレクトリと提供されている`train.csv`、`train_localizers.csv`を詳しく調べてください。
4.  **モデルを開発する：** 動脈瘤検出と局在化のための機械学習モデルを作成します。
5.  **ソリューションを提出する：** 提供された評価APIを利用してください。詳細な提出手順については、[こちらの例のノートブック](https://www.kaggle.com/code/awsaf49/mean-weighted-columnwise-aucroc)を参照してください。
6.  **知見を共有する：** 発見や改善点を共有するために、自由にissueを開いたり、プルリクエストを作成したりしてください！

</details>

---

## License | ライセンス

### English
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<details>
<summary>日本語訳を表示</summary>

このプロジェクトはMITライセンスの下でライセンスされています。詳細については[LICENSE](LICENSE)ファイルを参照してください。

</details>

---

### **Let's detect aneurysms before they rupture!**
### **破裂する前に動脈瘤を検出しましょう！**

---

---

### **表示されない場合のトラブルシューティング**

もしこのコードを`README.md`に貼り付けてもGitHub上で折りたたみが機能しない場合は、以下を確認してみてください。

* **GitHubにプッシュしましたか？** ローカルで編集しただけでは、GitHub上の表示は変わりません。`git add .` -> `git commit -m "update README"` -> `git push` の手順で変更を反映させてください。
* **ブラウザのキャッシュをクリアしてみる**：まれにブラウザのキャッシュが原因で古い表示が残っていることがあります。
* **GitHubのWebエディタで直接編集してみる**：もし可能であれば、GitHubのリポジトリページで`README.md`を直接編集し、プレビュー機能で動作を確認してみてください。

この完全なコードで、ご希望の表示が実現できることを願っています。他に何かお手伝いできることがありましたら、お気軽にお声がけください！
