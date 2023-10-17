# BioDNet

# 🔍 Overview 

BioDNet (Bioinformatics & Deep Neural Network Explorer) leverages the power of [OptNCMiner](https://github.com/phytoai/OptNCMiner), a tool rooted in deep neural networks. By seamlessly blending machine learning with bioinformatics, BioDNet pioneers in probing natural compounds to unveil groundbreaking associations with diseases. This aids researchers in delving deeper into the untapped potential of natural compounds in the realm of disease research.

# 📊 Collecting Natural Compound Data from FooDB 

This section provides a comprehensive guide and scripts for collecting natural compound data from FooDB, a valuable resource for nutritional and chemical information on various food compounds. The data collection process involves downloading the FooDB MySQL dump file, importing it into a database, and extracting relevant information, specifically the Simplified Molecular Input Line Entry System (SMILES) notation.

## 🚀 Getting Started 

Follow these steps to collect natural compound data from FooDB:

### 📥 Step 1: Download the FooDB MySQL Dump 

Visit the FooDB downloads page at [https://foodb.ca/downloads](https://foodb.ca/downloads) to access the latest FooDB MySQL dump file. Ensure you download the appropriate version.

### 🗃️ Step 2: Import the Dump File 

1. Decompress the downloaded FooDB MySQL dump file (`foodb_2020_4_7_mysql.tar.gz`).

2. Create a new database named `foodb` using the following SQL command:

   ```sql
   CREATE DATABASE foodb;
   

### Import the FooDB dump file into the newly created database
   
replacing [userid], [password], and {path/foodb_server_dump_2020_4_21}.sql with your MySQL credentials and the file path, respectively: mysql -u [userid] -p [password] < {path/foodb_server_dump_2020_4_21}.sql

### 📦 Step 3: Extract Data 

Access the foodb database:USE foodb;
Retrieve unique SMILES notations for compounds with export status and foods marked for export to FooDB:
   
     SELECT DISTINCT c.moldb_smiles
     FROM contents t
     JOIN compounds c ON t.source_id = c.id
     JOIN foods f ON t.food_id = f.id
     WHERE t.source_type = 'Compound' AND c.export = 1 AND f.export_to_foodb = 1;

Download the extracted data in CSV format, saved as moldb_smiles.csv.

### 👋 Step 4: Add fingerprint

```r
#install.packages('caret')
#install.packages('rcdk')
library(caret)
library(rcdk)

data<-read.csv("moldb_smiles.csv")
dim(data)
  
# extract SMILES column
data<-as.character(data[,1])
 
# parsing
mols = parse.smiles(data)

trans <- data.frame(data[!sapply(mols, is.null)], stringsAsFactors=F)
mols = parse.smiles(trans[,1])

fps = lapply(mols, get.fingerprint, type='standard')
fps.matrix = fingerprint::fp.factor.matrix(fps)

# save data
final <- data.frame(cbind(trans, fps.matrix))
write.csv(final, "foodb_compounds_fp.csv",row.names=FALSE)
```
# ✨ Other documentation of model training and interacting compounds analysis

**Model Definition**: Dive deep into the structure and intricacies of our neural network model.
   - [Read about our model here](https://github.com/Benjamin-JHou/DNNCDiscover/blob/main/Train_TestCode/model.md)

**Preprocessing**: Understand how we transform our raw data into a format suitable for model training and inference.
   - [Check out preprocessing steps here](https://github.com/Benjamin-JHou/DNNCDiscover/blob/main/Train_TestCode/preprocess.md)

**Utility Functions**: Various helper functions that ease our operations, be it file I/O, logging, or any other utility operations.
   - [Explore our utilities here](https://github.com/Benjamin-JHou/DNNCDiscover/blob/main/Train_TestCode/utils.md)

Detailed information and examples regarding the testbed utilized for model training and evaluation can be found in the 
   - [Testbed](https://github.com/Benjamin-JHou/DNNCDiscover/blob/main/Train_TestSet/testbed.md)

For further details on the Quantitative Structure-Activity Relationship (QSAR) analysis, please refer to the 
   - [QSAR documentation](https://github.com/Benjamin-JHou/DNNCDiscover/blob/main/myocardial%20infarction/preprocess_QSAR.md)

For more details on screening and predictive analysis of interacting compounds, see 
   - [prediction Documentation](https://github.com/Benjamin-JHou/DNNCDiscover/blob/main/myocardial%20infarction/prediction.md)


# ✅ Data Verification and Processing

Post-prediction of the potential natural compounds, a meticulous verification process was initiated by iterating over the Compound Identifiers (CIDs) to extract pertinent information. Particularly, it is imperative to note that the IC50 values were averaged for each natural compound, providing a single IC50 value per compound. Employing mean IC50 values is derived from consistent experimental setups rationale and support from literature. Averaging IC50 values could be considered a pragmatic approach, especially when multiple measurements or replicates are available.

   ```python
    # Load the data
    try:
    df = pd.read_csv('NC_CID.csv')  # replace with your filename
    except FileNotFoundError:
    print("File not found. Please check your filename and path.")
    raise

    # Initialize a DataFrame to store bioactivity data
    bioactivity_df = pd.DataFrame()

    # Loop through CIDs in the data
    for index, row in df.iterrows():
    cid = str(row['CID'])
    name = row['Name']
    
    print(f"Fetching data for CID {cid}, Name {name}...")
    
    # Construct the URL for the API request
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/assaysummary/JSON"
    
    # Get the data
    try:
        response = requests.get(url)
        response.raise_for_status()  # check if the request was successful
        data = response.json()
    except requests.RequestException as e:
        print(f"Failed to fetch data for CID {cid}, Name {name}. Error: {str(e)}")
        continue
    
    # Extract relevant information
    try:
        bioactivity_data = data['Table']['Row']
        temp_df = pd.DataFrame(bioactivity_data)
        temp_df['Name'] = name  # add a column with the compound name
        temp_df['CID'] = cid    # add a column with the CID
        
        # Append to the main DataFrame
        bioactivity_df = pd.concat([bioactivity_df, temp_df], ignore_index=True)
    except KeyError:
        print(f"No bioactivity data found for CID {cid}, Name {name}.")
        continue
    
    print(f"Data fetched for CID {cid}, Name {name}.")

    # Save the bioactivity data to a CSV file
    bioactivity_df.to_csv('bioactivity_data.csv', index=False)
    print("Bioactivity data saved to bioactivity_data.csv.")
```
## ✍️ Compound Clustering
```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

# data
df = pd.read_csv('bioactivity_data.csv')
smiles = df['Smiles']
ic50_values = df['IC50']

# Convert IC50 to pIC50
pic50_values = [-np.log10(value * 1e-9) for value in ic50_values]
pic50_standardized = StandardScaler().fit_transform(np.array(pic50_values).reshape(-1, 1))

# Generate fingerprints
fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=1024) for smi in smiles]
fp_matrix = [list(fp) for fp in fps]

# Combine fingerprints and standardized pIC50 values
combined_data = [fp + list(pic50) for fp, pic50 in zip(fp_matrix, pic50_standardized)]

# K-Means clustering
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(combined_data)

# Evaluate clustering
labels = kmeans.labels_
silhouette_avg = silhouette_score(combined_data, labels)
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Visualize using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(combined_data)

# Extracting explained variance
explained_variance = pca.explained_variance_ratio_

colors = ['black' if label == 0 else 'orange' for label in labels]
edge_colors = ['orange' if color == 'black' else 'black' for color in colors]

plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, edgecolors=edge_colors, s=50, linewidth=1)
plt.title('PCA visualization of clusters')
plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
plt.grid(linestyle='--')

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='1', markersize=10, markerfacecolor='orange', markeredgewidth=1.5, markeredgecolor='black'),
                   Line2D([0], [0], marker='o', color='w', label='2', markersize=10, markerfacecolor='black', markeredgewidth=1.5, markeredgecolor='orange')]
plt.legend(handles=legend_elements, title="Clusters", loc='upper right')
plt.savefig('cluster.png', dpi=300, bbox_inches='tight')
plt.show()
```
## ✍️ Descriptive statistics: Representativeness of compounds
```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS

# 1. Descriptive Statistics, calculate descriptive statistics for pIC50 for each cluster, such as mean, median, standard deviation
df['Cluster'] = labels
df_grouped = df.groupby('Cluster')['IC50'].describe()
print(df_grouped)

# 2. Representativeness of compounds, select the compound closest to the cluster center in each cluster as a representative
cluster_centers = kmeans.cluster_centers_
representatives = {}

for i, center in enumerate(cluster_centers):
    distances = np.linalg.norm(combined_data - center, axis=1)
    cluster_data = df[df['Cluster'] == i]
    median_index = cluster_data.iloc[np.argsort(distances)[len(distances)//2]].name
    representatives[i] = df.loc[median_index, 'Name']

print(representatives)

# Descriptor-based analysis of structural differences between clusters
from rdkit.Chem import Descriptors

# Calculate molecular weight for all compounds
df['MolWt'] = df['Smiles'].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))

# Compare the distribution of molecular weights across clusters
grouped = df.groupby('Cluster')['MolWt'].describe()
print(grouped)
```
## ✍️ Heat map: Distribution of fingerprint bits of molecules in different clusters
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.colors as mcolors

df = pd.read_csv('bioactivity_data.csv')

# 1. Generate fingerprints
df['Fingerprint'] = df['Smiles'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024))

# 2. Calculate the frequency of each bit in each cluster
df['Cluster'] = labels
clusters = df['Cluster'].unique()
bit_frequencies = []

for cluster in clusters:
    cluster_fps = df[df['Cluster'] == cluster]['Fingerprint'].tolist()
    avg_fp = np.mean(cluster_fps, axis=0)
    bit_frequencies.append(avg_fp)

bit_frequencies = np.array(bit_frequencies)

# 3.heat map drowing
plt.figure(figsize=(15, len(clusters)))
colors = ["black", "orange", "red"]
cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)

sns.heatmap(bit_frequencies, cmap=cmap, cbar_kws={'label': 'Bit Presence Frequency'})
plt.xlabel("Fingerprint Bits")
plt.ylabel("Clusters")
plt.yticks(ticks=np.arange(0.5, len(clusters)), labels=['1', '2'], rotation=0)  # Set the ytick labels to 1 and 2
plt.title("Fingerprint Bit Distribution Across Clusters")
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```
## ✍️ compute Maximum Common Substructure (MCS) for activity level
```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS, Draw
import matplotlib.pyplot as plt
from PIL import Image
import io
from IPython.display import display

# Load data
data = pd.read_csv('bioactivity_data.csv')

# Calculate pIC50 and categorize activity
data['pIC50'] = -np.log10(data['IC50'] * 1e-9)
data['Activity_Level'] = pd.cut(data['pIC50'], bins=[-np.inf, 5, 7, np.inf], labels=['Low', 'Medium', 'High'])

activity_levels = ['Low', 'Medium', 'High']

fig, axs = plt.subplots(1, len(activity_levels), figsize=(15, 5), sharex=True, sharey=True)

for ax, level in zip(axs, activity_levels):
    subset = data[data['Activity_Level'] == level]
    subset_mols = [Chem.MolFromSmiles(smiles) for smiles in subset['SMILES']]
    
    if len(subset_mols) > 1:
        mcs = rdFMCS.FindMCS(subset_mols)
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)

        # Generate high-quality image
        img = Draw.MolsToGridImage(
            subset_mols, 
            molsPerRow=4, 
            subImgSize=(600, 600),  # Increased image size for better quality
            legends=subset['Name'].tolist(), 
            highlightAtomLists=[mol.GetSubstructMatch(mcs_mol) for mol in subset_mols],
            useSVG=False
        )

        # Convert to PIL Image for more control over display settings
        img_pil = Image.open(io.BytesIO(img.data))
        
        # Display and save image
        ax.imshow(img_pil)
        ax.axis('off')
        ax.set_title(f'MCS for {level}')
        
        # Save with high dpi
        img_pil.save(f"MCS_{level}.png", dpi=(400, 400))
    else:
        print(f"Not enough molecules to compute MCS for activity level {level}.")
        ax.axis('off')
        ax.set_title(f'Insufficient data for {level}')

plt.tight_layout()
plt.show()
```


# ✍️ Citation 

If you use data collected from FooDB in your research, please consider citing FooDB as the data source in your publications. You can find citation information on the FooDB website.

# 📄 License 

This repository is provided under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html), which allows you to:

- Use the code for any purpose, including research and commercial projects.
- Modify and distribute the code, provided that derivative works are also released under the GPL-3.0 license.

For more information about the GPL-3.0 license and its permissions and restrictions, please refer to the [official GNU GPL-3.0 page](https://www.gnu.org/licenses/gpl-3.0.en.html).

For more information and updates, please refer to the official FooDB website: [https://foodb.ca/](https://foodb.ca/)

