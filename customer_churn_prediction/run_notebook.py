import nbformat
from nbclient import NotebookClient

# Load the notebook
with open("Customer_Churn_Prediction.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)
    
# Execute the notebook
print("Executing notebook...")
client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

# Save it back
with open("Customer_Churn_Prediction.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
    
print("Notebook executed and saved successfully.")
