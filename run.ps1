cls

#RUN QDRANT
#docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# ACTIVATE VENV AND RUN
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r ./requirements.txt

cls

python ./main.py
