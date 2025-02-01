
# **Market Data Pipeline (`market_data`)**
🚀 A standalone project for retrieving, storing, and updating **historical & daily market data** for **equities, ETFs, and cryptocurrencies** using **Tiingo API**.

## **📌 Features**
- ✅ **Retrieves historical & daily market data** for equities, ETFs, and cryptocurrencies.
- ✅ **Automatically detects new tickers** and fetches their metadata.
- ✅ **Supports incremental updates** (only fetches new data).
- ✅ **Stores data in PostgreSQL** with structured tables.
- ✅ **Uses YAML for easy asset management.**
- ✅ **Automated with `cron` for daily updates.**

---
## **📂 Project Structure**
```
market_data/
│── data_retrieval.py   # Main script to fetch and store market data
│── macro_assets.yaml   # List of tracked equities & crypto (configurable)
│── .env                # API key (ignored by Git)
│── database.ini        # Database connection config (ignored by Git)
│── README.md           # This documentation
│── db_utils.py         # Database interaction functions
│── db_config.py        # Database configuration
│── requirements.txt    # Required Python packages
```

---

## **🔧 Setup Instructions**
### **1️⃣ Install Dependencies**
```
pip install -r requirements.txt
```

Provide Required Config Files
Ensure the following files exist in the project root:

✅ .env (API Key)
Create a file named .env and add:
```
TIINGO_API_KEY=my_tiingo_api_key
```
✅ database.ini (PostgreSQL Connection)
Create a file named database.ini with:
```
[postgresql]
host=your_db_host
database=your_db_name
user=your_db_user
password=your_db_password
port=your_db_port
```
🚀 Running the Script
Use the following command to run data_retrieval.py:

`eval $(cat .env) PYTHONPATH=. python data_retrieval.py`

Or specify a custom asset file:

`eval $(cat .env) PYTHONPATH=. python data_retrieval.py --assets_file=my_assets.yaml`

⚙ Automating with Cron Job
To schedule daily updates, add this line to crontab -e:

`0 0 * * * cd /path/to/market_data && eval $(cat .env) PYTHONPATH=. python data_retrieval.py`