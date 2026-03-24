# Urban Heat Island Prediction Model

This project includes:
- A structured Jupyter notebook with synthetic dataset generation (1000 rows), EDA, model training, and export.
- A Flask backend with `/predict` endpoint.
- A frontend (HTML/CSS/JS) for interactive predictions.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Project structure

- `notebooks/urban_heat_island_model.ipynb` — EDA and ML workflow
- `src/train_model.py` — synthetic data + model training utility
- `app.py` — Flask app
- `templates/index.html` — frontend page
- `static/css/styles.css` — styling
- `static/js/app.js` — prediction request logic
