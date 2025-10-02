cls

#RUN QDRANT
#docker run -d -p 6333:6333 qdrant/qdrant

# ACTIVATE VENV AND RUN
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r ./requirements.txt

cls

python ./main.py
