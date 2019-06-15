python tools/create_dictionary.py
python tools/compute_softscore.py

cd data
python create_vqacp_dataset.py
python create_vqx_dataset.py
python create_vqx_hint.py
cd ..
