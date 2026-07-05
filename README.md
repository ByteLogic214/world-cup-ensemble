## World Cup Ensemble Predictor

Modelo ensemble para predicción 1X2 en fútbol usando:
- API-Football para históricos
- The Odds API para cuotas
- RandomForest + XGBoost + LightGBM + LogisticRegression
- Soft voting calibrado

## Secrets en GitHub
Configura estos secrets en tu repositorio:
- API_FOOTBALL_KEY
- ODDS_API_KEY

## Ejecutar desde GitHub Actions
1. Sube estos archivos al repositorio.
2. Ve a **Settings > Secrets and variables > Actions**.
3. Crea `API_FOOTBALL_KEY`.
4. Crea `ODDS_API_KEY`.
5. Ve a **Actions**.
6. Ejecuta `Train World Cup Ensemble`.

## Estructura
- `src/main.py`: pipeline de entrenamiento
- `src/api_clients.py`: clientes API
- `src/features.py`: ingeniería de variables
- `src/model.py`: ensemble calibrado
- `.github/workflows/train.yml`: entrenamiento automático

## Salidas
- `data/raw/fixtures.csv`
- `data/raw/odds.csv`
- `data/processed/match_feat
