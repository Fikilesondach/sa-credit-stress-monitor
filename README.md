# 🇿🇦 SA Credit Stress Monitor

> **A real-time South African credit risk and macroeconomic stress-testing dashboard built with Streamlit, powered by live market and economic data.**

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://sa-credit-stress-monitor-fsc7e8mz5xcqmmyxepn3dg.streamlit.app)
[![GitHub](https://img.shields.io/badge/Source-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Fikilesondach/sa-credit-stress-monitor)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

---

## 📌 Overview

The **SA Credit Stress Monitor** is an interactive analytical dashboard designed to monitor, model, and stress-test credit risk conditions in the South African economy. It integrates live macroeconomic data from sources such as the World Bank and FRED, alongside market data from Yahoo Finance, to provide a dynamic view of credit stress indicators relevant to South Africa.

The dashboard is intended for analysts, researchers, risk practitioners, and students who want to explore how macroeconomic shocks — such as interest rate changes, unemployment spikes, or currency depreciation — propagate into credit risk outcomes.

---

## 🚀 Live Demo

👉 **[Launch the App](https://sa-credit-stress-monitor-fsc7e8mz5xcqmmyxepn3dg.streamlit.app)**

No installation required. The app runs entirely in your browser.

---

## ✨ Features

- **Real-time macroeconomic data** — pulls live indicators from the World Bank (`wbdata`), FRED (`fredapi`), and Yahoo Finance (`yfinance`)
- **Credit stress indicators** — tracks key metrics relevant to the South African credit environment, including interest rates, unemployment, exchange rates, and more
- **Scenario stress testing** — model the impact of adverse macroeconomic shocks on credit risk
- **Interactive visualisations** — rich, dynamic charts powered by Plotly
- **Clean, lightweight dashboard** — built with Streamlit for rapid, browser-based interaction
- **Modular architecture** — dashboard is decoupled from the ML training stack, keeping deployment lean and conflict-free

---

## 🗂️ Project Structure

```
sa-credit-stress-monitor/
├── dashboard/
│   └── app.py                  # Main Streamlit application entry point
├── requirements.txt            # Dashboard dependencies (lean, deployment-ready)
├── runtime.txt                 # Python version pin (3.11)
├── packages.txt                # System-level apt dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Dashboard framework | [Streamlit](https://streamlit.io/) |
| Data visualisation | [Plotly](https://plotly.com/python/) |
| Data manipulation | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| Macroeconomic data | [wbdata](https://github.com/OliverSherouse/wbdata) (World Bank), [fredapi](https://github.com/mortada/fredapi) (FRED) |
| Market data | [yfinance](https://github.com/ranaroussi/yfinance) |
| Environment config | [python-dotenv](https://github.com/theskumar/python-dotenv) |
| HTTP client | [requests](https://docs.python-requests.org/) |
| Python version | 3.11 |
| Deployment | [Streamlit Community Cloud](https://streamlit.io/cloud) |

---

## ⚙️ Local Setup

### Prerequisites

- Python 3.11
- `git`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Fikilesondach/sa-credit-stress-monitor.git
cd sa-credit-stress-monitor

# 2. Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure environment variables
cp .env.example .env
# Edit .env with your API keys (e.g. FRED API key)
```

### Running the App

```bash
streamlit run dashboard/app.py
```

The app will open at `http://localhost:8501`.

---

## 🔑 Environment Variables

If the app uses API keys (e.g. for FRED), create a `.env` file in the project root:

```env
FRED_API_KEY=your_fred_api_key_here
```

You can obtain a free FRED API key at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html).

For Streamlit Cloud deployments, add secrets via the **Manage App → Secrets** panel instead of a `.env` file.

---

## ☁️ Deployment

The app is deployed on **Streamlit Community Cloud** and triggered automatically on pushes to the `main` branch.

Key deployment files:

- `runtime.txt` — pins the Python runtime to `python-3.11` to ensure package compatibility
- `requirements.txt` — intentionally lean (dashboard-only dependencies; no ML training stack) to avoid dependency conflicts on the cloud environment
- `packages.txt` — any system-level apt packages required

> **Note:** If you fork this repo and deploy your own instance, ensure both `runtime.txt` and `requirements.txt` are committed to the **repository root** (not inside the `dashboard/` subfolder), as Streamlit Cloud reads them from there.

---

## 📊 Data Sources

| Source | Data Provided |
|---|---|
| [World Bank (wbdata)](https://data.worldbank.org/) | GDP growth, inflation, unemployment, credit indicators |
| [FRED (fredapi)](https://fred.stlouisfed.org/) | US and global interest rate benchmarks, financial stress indices |
| [Yahoo Finance (yfinance)](https://finance.yahoo.com/) | ZAR exchange rates, equity indices, bond yields |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 📄 License

This project is open source. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Fikile Sondach**
- GitHub: [@Fikilesondach](https://github.com/Fikilesondach)

---

*Built with ❤️ for the South African financial analytics community.*
