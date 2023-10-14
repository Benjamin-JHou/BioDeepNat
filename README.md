# DNNCDiscover

# Overview 🔍

DNNCDiscover is a deep neural network-based tool designed for exploring natural compounds to discover novel associations with diseases. It combines machine learning and bioinformatics approaches, aiding researchers in gaining deeper insights into the potential of natural compounds in disease research.

# Collecting Natural Compound Data from FooDB 📊

This section provides a comprehensive guide and scripts for collecting natural compound data from FooDB, a valuable resource for nutritional and chemical information on various food compounds. The data collection process involves downloading the FooDB MySQL dump file, importing it into a database, and extracting relevant information, specifically the Simplified Molecular Input Line Entry System (SMILES) notation.

## Getting Started 🚀

Follow these steps to collect natural compound data from FooDB:

### Step 1: Download the FooDB MySQL Dump 📥

Visit the FooDB downloads page at [https://foodb.ca/downloads](https://foodb.ca/downloads) to access the latest FooDB MySQL dump file. Ensure you download the appropriate version.

### Step 2: Import the Dump File 🗃️

1. Decompress the downloaded FooDB MySQL dump file (`foodb_2020_4_7_mysql.tar.gz`).

2. Create a new database named `foodb` using the following SQL command:

   ```sql
   CREATE DATABASE foodb;
   

### Import the FooDB dump file into the newly created database
   
replacing [userid], [password], and {path/foodb_server_dump_2020_4_21}.sql with your MySQL credentials and the file path, respectively: mysql -u [userid] -p [password] < {path/foodb_server_dump_2020_4_21}.sql

### Step 3: Extract Data 📦

Access the foodb database:USE foodb;
Retrieve unique SMILES notations for compounds with export status and foods marked for export to FooDB:
   
     SELECT DISTINCT c.moldb_smiles
     FROM contents t
     JOIN compounds c ON t.source_id = c.id
     JOIN foods f ON t.food_id = f.id
     WHERE t.source_type = 'Compound' AND c.export = 1 AND f.export_to_foodb = 1;

Download the extracted data in CSV format, saved as moldb_smiles.csv.

# Data Verification and Processing

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


## Citation ✍️

If you use data collected from FooDB in your research, please consider citing FooDB as the data source in your publications. You can find citation information on the FooDB website.

## License 📄

This repository is provided under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html), which allows you to:

- Use the code for any purpose, including research and commercial projects.
- Modify and distribute the code, provided that derivative works are also released under the GPL-3.0 license.

For more information about the GPL-3.0 license and its permissions and restrictions, please refer to the [official GNU GPL-3.0 page](https://www.gnu.org/licenses/gpl-3.0.en.html).

For more information and updates, please refer to the official FooDB website: [https://foodb.ca/](https://foodb.ca/)

