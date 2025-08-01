# 🛍️ Customer Segmentation for E-Commerce
Segment customer based on purchasing behavior

# 🔗 Streamlit App
https://customersegmentatione-commerce-amjqnfabzeuumzntsezvyp.streamlit.app/

# 🧠 Project Structure
```plaintext
📁 customer_segmentation_app/
│
├── 📄 app.py                 # Main Streamlit app
├── 📁 data/                  # Folder for uploade data
│   └── customers.csv
├── 📁 modules/               # Python modules for logic separation
│   ├── eda.py                
│   ├── clustering.py         
│   └── visualization.py      
├── 📄 requirements.txt        # Required libraries
└── 📄 README.md               # Project description
```
# 📁 Dataset
```plaintext
This is a transactional data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.
The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
```
```plaintext
This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.
```
# 📊 Features
## 1. EDA (Exploratory Data Analysis)
```plaintext
1. Summary statistics of key metrics (Quantity, Unit Price, Total Price)
2. Top 10 products by quantity and revenue
3. Sales by country
4. Correlation heatmap of features
```
## 2. Customer Segmentation
```plaintext
1. RFM (Recency, Frequency, Monetary) analysis
2. K-Means clustering with adjustable number of clusters (K)
3. Cluster summary table with customer count, suggest a brief marketing strategy by Gemini AI
4. Downloadable CSV of all clustered customers
5. Elbow Method and Silhouette Score visualizations for optimal K selection
```
## 3. 3D Cluster Visualization
```plaintext
1. Interactive 3D scatter plots of customer clusters using:
- RFM dimensions
- PCA (Principal Component Analysis)
- t-SNE (t-distributed Stochastic Neighbor Embedding)
2. Visualized using Plotly for intuitive exploration
```
