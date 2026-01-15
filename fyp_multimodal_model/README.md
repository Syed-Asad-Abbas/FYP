
# FYP Multimodal Phishing Models (Prototype)

This repo gives you **three separate, runnable models** matching your modalities:

1) **URL → LightGBM** (`train_url_lightgbm.py`)
2) **DOM → Doc2Vec + LightGBM** (`train_dom_doc2vec_lgbm.py`)
3) **Visual → ResNet50 (transfer learning)** (`train_visual_resnet.py`)

All scripts read the provided dataset: `/mnt/data/PhiUSIIL_Phishing_URL_Dataset.csv`.

> **Labels**: assumes `label` column is `1 = phishing`, `0 = benign`.

---

## Quickstart

```bash
cd /mnt/data/fyp_multimodal_model
python -m pip install -r requirements.txt

# URL model
python train_url_lightgbm.py --config config.json

# DOM model
python train_dom_doc2vec_lgbm.py --config config.json --vector_size 64 --epochs 30

# Visual model
# Put screenshots in: /mnt/data/fyp_multimodal_model/screenshots/<FILENAME>.png
python train_visual_resnet.py --config config.json --epochs 5
```

### Where outputs go?
- Models & metrics: `/mnt/data/fyp_multimodal_model/models`
  - `url_lgbm.joblib`, `url_metrics.json`
  - `dom_doc2vec_lgbm.joblib`, `dom_metrics.json`
  - `visual_resnet50.pt`, `visual_metrics.json`

---

## Notes

- **URL model** uses only the lexical/statistical columns available in the CSV (no external WHOIS calls). If you later add WHOIS features (e.g., domain_age_days), the script will automatically pick them if you add the column name to the list.
- **DOM model** generates a *synthetic DOM text* per row from DOM-related booleans (like `HasPasswordField`) and counts (`NoOfImage`, `NoOfJS`, ...). We then train **Doc2Vec** to get embeddings and classify them via **LightGBM** – matching your requested approach in spirit using the dataset you have.
- **Visual model** requires screenshots. The CSV includes `FILENAME`; if you export screenshots named `<FILENAME>.png` to `/mnt/data/fyp_multimodal_model/screenshots`, the script will use them. We fine‑tune the **ResNet50 head** for binary phishing classification. If no images are present, the script will tell you.
