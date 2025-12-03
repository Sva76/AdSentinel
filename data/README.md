# Data

This folder is *not* versioned with the GDPa1 data (because of size and license).

Download the GDPa1 dataset from the Ginkgo / Hugging Face space and place here:

- `GDPa1_v1.2_20250814.csv` (or the official CSV from the challenge)
- `heldout-set-sequences.csv`

Then you can run:

```bash
python -m adsentinel.train_cv --train-csv data/GDPa1_v1.2_20250814.csv --out-csv outputs/gdpa1_cv_predictions.csv

python -m adsentinel.predict \
  --train-csv data/GDPa1_v1.2_20250814.csv \
  --heldout-csv data/heldout-set-sequences.csv \
  --out-train-csv outputs/gdpa1_train_preds.csv \
  --out-heldout-csv outputs/gdpa1_heldout_preds.csv

Commit.

---

## 7️⃣ Aggiornare il README con “Installation / Usage”

Ora puoi cliccare su **README.md → Edit** e sotto “Repository structure” aggiungi, ad esempio:

```markdown
## Installation

```bash
git clone https://github.com/Sva76/AdSentinel.git
cd AdSentinel
python -m venv .venv
source .venv/bin/activate  # su Windows: .venv\Scripts\activate
pip install -r requirements.txt
