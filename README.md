# Detecting Genre Bias in Tweets

## Authors
- Ana Pinto up202105085 
- Pedro Leitão up202107852
- Pedro Oliveira up202308236

---

This project investigates **genre bias in tweet annotations** by analyzing annotator behavior and developing predictive models using the EXIST 2025 dataset. Our primary goal is to understand how demographic characteristics influence labeling decisions and to create systems that account for or mitigate potential biases in annotation.

We employ various techniques from machine learning, natural language processing (NLP), and data mining to achieve these objectives, including clustering, association rule mining, and recommendation frameworks.

---

## 🧠 Objectives

- **Identify** patterns in annotator labeling behavior.
- **Explore** demographic influences on annotation decisions.
- **Develop** models to predict or adjust for annotation bias.
- **Visualize** relationships between annotator traits and tweet classifications.

---

## 📁 Project Structure

### 🔍 Analytical Tasks

- **`Task3.R`** → *Annotator Clustering*  
  Clusters annotators based on labeling behavior using unsupervised learning. This helps uncover trends among annotators, such as consistent over- or under-labeling of bias, and explores relationships with demographic features (e.g., age, gender, country).

- **`Task4.R`** → *Association Rule Mining*  
  Uses the Apriori algorithm to mine interesting rules that associate annotator demographics with their annotation patterns. For example, it uncovers how specific combinations of traits correlate with a higher likelihood of marking a tweet as biased.

- **`Task5.R`** → *Recommendation System & Advanced Analysis*  
  Builds on the insights from Tasks 3 and 4 to propose recommendation-style or predictive models. These models aim to predict likely annotations based on annotator profiles or to recommend annotation adjustments.

- **`code_task2.R`** → *Experimental Task 2*  
  Early-stage or experimental scripts related to feature engineering and preliminary analysis. *(To be updated or deprecated in future versions.)*


### ⚙️ Additional Files

- **`diagram.md`** → *Visual Summary*  
  Contains diagrams that illustrate the data pipeline, modeling workflow, and key findings.

---

## 📦 Data Directory

The `data/` folder contains all essential datasets used throughout the project:

- **`EXIST2025_train.csv`** → Primary dataset for training models, includes tweets and annotations from multiple annotators.
- **`EXIST2025_dev_labeled.csv`** → Development dataset with known labels for validation and testing.
- **`EXIST2025_dev_unlabeled.csv`** → Unlabeled set for exploratory analysis or pseudo-labeling approaches.

---

## 🧪 Technical Details

- **Language**: R
- **Libraries**: `tidyverse`, `cluster`, `arules`, `ggplot2`, and other relevant statistical/NLP packages
- **Techniques**: Clustering, Association Rule Mining, Recommender Systems, Feature Engineering, Demographic Analysis

---

## 💾 Environment & Session Files

- **`.RData`** and **`.Rhistory`**  
  R session and history files preserving environment variables and command history for reproducibility.

---