import os
from sec_edgar_downloader import Downloader

# SEC EDGAR requires a user-agent header
dl = Downloader("AirlineAnalyticsPOC", "your_email@example.com", "sec_data")

tickers = ["UAL", "DAL", "AAL", "LUV"]
filings = ["10-K", "10-Q", "8-K"]

for ticker in tickers:
    for filing in filings:
        print(f"Fetching {filing} for {ticker} up to 2026...")
        dl.get(
            filing,
            ticker,
            after="2019-01-01",
            before="2026-01-01",
            download_details=True
        )

print("SEC filings downloaded successfully!")
