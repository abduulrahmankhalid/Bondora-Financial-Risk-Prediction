# exit on error
set -o errexit

pip install --upgrade pip
pip install gunicorn

pip install scikit-learn

pip install -r requirements.txt
