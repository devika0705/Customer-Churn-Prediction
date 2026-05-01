import json
try:
    with open('create_notebook.py', 'r', encoding='utf-8') as f:
        txt = f.read()
    txt = txt.replace("sns.countplot(context='notebook', data=df, x='Churn')", "sns.countplot(data=df, x='Churn')")
    with open('create_notebook.py', 'w', encoding='utf-8') as f:
        f.write(txt)
    print("Fixed create_notebook.py")
except Exception as e:
    print(e)
