# 📝 **Explanation for preprocess.py** 📝

🔍 **Overview:**
This Python script, `preprocess.py`, is designed to preprocess data for machine learning tasks, focusing primarily on the manipulation of multi-label datasets. It offers functionalities such as splitting datasets, converting multi-label to single-label, preprocessing data from directories, and more.�

## 🚀 **Main Functions:**

1. 🌌 **preprocess_multilabel()** 
    - Objective: Processes a multi-label dataset.
    - Features:
        - Reads a train.csv file and splits it into training and testing sets.
        - Converts multi-label data to single label.
        - Creates positive and negative examples from the dataset.
        - Logs the progress.

2. 🏷 **toSingleLabel()** 
    - Objective: Converts multi-label data to single-label data.
    - Features:
        - Reads a specified file and isolates its label columns.
        - Randomly selects a single label for each row.
        - Saves the modified data back to the file.

3. 📂 **preprocess_directory()** 
    - Objective: Processes all CSV files in a directory.
    - Features:
        - Reads all CSV files within a specified path.
        - Merges and shuffles datasets.
        - Performs stratified sampling.
        - Logs the progress.

4. 🔄 **preprocess()** 
    - Objective: Generates positive and negative examples.
    - Features:
        - For positive examples: Picks two rows with the same label.
        - For negative examples: Picks two rows with different labels.
        - Logs the progress.

5. 🧱 **addSupport()** 
    - Objective: Augments the dataset with support data.
    - Features:
        - Reads all CSV files within a specified path.
        - Extracts rows that match a specified condition.
        - Divides rows into supporting data and test data.

## 🔧 **Execution:**
To run the script, execute it directly. The logging settings are initialized in the main body, which logs information at the INFO level.

## 🎉 **Wrapping Up:**
The script is designed to be modular and can be easily integrated with other machine learning pipelines. The extensive logging ensures that the user is kept informed of the preprocessing steps and progress.
