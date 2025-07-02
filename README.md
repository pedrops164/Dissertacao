# Steps to run

# Create venv environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install packages (when activated)
pip install -r requirements.txt

# Pull the Qdrant Docker Image
docker pull qdrant/qdrant

# Create directory to store qdrants data:
mkdir -p ./qdrant_storage

# Run qdrant db container
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

# Populate qdrant db
python populate_db.py --num-docs 100000 --use-quantization --n-workers 8

# Run evaluation
python evaluation.py