# Standard library imports
import json
import time
import logging
from functools import lru_cache

# Third-party imports
import yfinance as yf
from colorama import Fore
import chromadb

# Local imports
from mcp.server.fastmcp import FastMCP

# Set up logging
log_file_path = "stock_server.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()  # This will also output logs to console
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {log_file_path}")

# Create server 
mcp = FastMCP("StockAnalysisServer", description="A server for analysing stock data using Yahoo Finance and LLMs")

# Add in a prompt function
@mcp.prompt()
def stock_summary(stock_data:str) -> str:
    """Prompt template for summarising stock price"""
    return f"""You are a helpful financial assistant designed to summarise stock data.
                Using the information below, summarise the pertinent points relevant to stock price movement
                Data {stock_data}"""

# ChromaDB initialization with error handling
def get_chroma_collection():
    try:
        chroma_client = chromadb.PersistentClient(path="ticker_db")
        return chroma_client.get_collection(name="stock_tickers")
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}")
        return None

# Add in a resource function
@mcp.resource("tickers://search/{stock_name}")
def list_tickers(stock_name:str)->str: 
    """This resource allows you to find a stock ticker by passing through a stock name e.g. Google, Bank of America etc. 
        It returns the result from a vector search using a similarity metric. 
    Args:
        stock_name: Name of the stock you want to find the ticker for
        Example payload: "Procter and Gamble"

    Returns:
        str:"Ticker: Last Price" 
        Example Response 
        {'ids': [['41', '30']], 'embeddings': None, 'documents': [['AZN - ASTRAZENECA PLC', 'NVO - NOVO NORDISK A S']], 'uris': None, 'included': ['metadatas', 'documents', 'distances'], 'data': None, 'metadatas': [[None, None]], 'distances': [[1.1703131198883057, 1.263759970664978]]}
        
    """
    try:
        collection = get_chroma_collection()
        if not collection:
            return "Error: Unable to connect to ticker database"
        results = collection.query(query_texts=[stock_name], n_results=1)
        return str(results)
    except Exception as e:
        logger.error(f"Error in list_tickers for {stock_name}: {e}")
        return f"Error searching for ticker: {str(e)}"

# Cache for ticker data to reduce API calls
@lru_cache(maxsize=128)
def get_ticker_data(ticker):
    """Fetches ticker data from Yahoo Finance and caches it.
    Args:
        ticker: Stock ticker symbol as a string, e.g., "AAPL" for Apple Inc.
    Returns:
        yf.Ticker: A yfinance Ticker object containing stock data.
    """
    logger.info(f"Fetching data for ticker: {ticker}")
    ticker = ticker.upper()  # Ensure ticker is uppercase
    try:
        tickerData = yf.Ticker(ticker)
        logger.info(f"Successfully fetched data for ticker: {ticker} - {tickerData.info.get('longName', 'No name available')}")
        return tickerData
    except Exception as e:
        logger.error(f"Error fetching ticker data for {ticker}: {e}")
        return None


# Build server function
@mcp.tool()
def stock_price(stock_ticker: str) -> str:
    """This tool returns the last known price for a given stock ticker.
    Args:
        stock_ticker: a alphanumeric stock ticker 
        Example payload: "NVDA"

    Returns:
        str:"Ticker: Last Price" 
        Example Response "NVDA: $100.21" 
        """
    try:
        # Specify stock ticker 
        dat = get_ticker_data(stock_ticker)
        # Get historical prices
        historical_prices = dat.history(period='1mo')
        # Filter on closes only
        last_months_closes = historical_prices['Close']
        logger.info(f"Retrieved prices for {stock_ticker}: {last_months_closes}")
        return str(f"Stock price over the last month for {stock_ticker}: {last_months_closes}")
    except Exception as e:
        logger.error(f"Error retrieving stock price for {stock_ticker}: {e}")
        time.sleep(1)  # Wait before retrying
        try:
            dat = get_ticker_data(stock_ticker)
            historical_prices = dat.history(period='1mo')
            last_months_closes = historical_prices['Close']
            return str(f"Stock price over the last month for {stock_ticker}: {last_months_closes}")
        except Exception as e2:
            return f"Error retrieving stock price for {stock_ticker}: {str(e2)}"

# Add in a stock info tool 
@mcp.tool()
def stock_info(stock_ticker: str) -> str:
    """This tool returns information about a given stock given it's ticker.
    Args:
        stock_ticker: a alphanumeric stock ticker
        Example payload: "IBM"

    Returns:
        str:information about the company
        Example Response "Background information for IBM: {'address1': 'One New Orchard Road', 'city': 'Armonk', 'state': 'NY', 'zip': '10504', 'country': 'United States', 'phone': '914 499 1900', 'website': 
                'https://www.ibm.com', 'industry': 'Information Technology Services',... }" 
        """
    try:
        # Specify stock ticker
        logger.info(f"Retrieving info for {stock_ticker}")
        dat = get_ticker_data(stock_ticker)
        if not dat.info:
            raise ValueError(f"No information found for ticker {stock_ticker}")
        logger.debug(f"Retrieved raw info for {stock_ticker}: {dat.info}")
        # Extract only the most relevant information
        relevant_info = {k: dat.info.get(k) for k in ['shortName', 'longName', 'sector', 'industry', 
                                                     'website', 'market', 'marketCap', 'country',
                                                     'city', 'state', 'zip', 'phone'] 
                        if k in dat.info}
        logger.info(f"Retrieved filtered info for {stock_ticker}: {relevant_info}")
        return f"Background information for {stock_ticker}: {json.dumps(relevant_info, indent=2)}"
    except Exception as e:
        logger.error(f"Error retrieving stock info for {stock_ticker}: {e}")
        time.sleep(1)  # Wait before retrying
        try:
            dat = get_ticker_data(stock_ticker)
            relevant_info = {k: dat.info.get(k) for k in ['shortName', 'longName', 'sector', 'industry', 
                                                         'website', 'market', 'marketCap', 'country',
                                                         'city', 'state', 'zip', 'phone'] 
                            if k in dat.info}
            return f"Background information for {stock_ticker}: {json.dumps(relevant_info, indent=2)}"
        except Exception as e2:
            return f"Error retrieving stock info for {stock_ticker}: {str(e2)}"

# Add in an income statement tool
@mcp.tool()
def income_statement(stock_ticker: str) -> str:
    """This tool returns the quarterly income statement for a given stock ticker.
    Args:
        stock_ticker: a alphanumeric stock ticker
        Example payload: "BOA"

    Returns:
        str:quarterly income statement for the company
        Example Response "Income statement for BOA: 
        Tax Effect Of Unusual Items                           76923472.474289  ...          NaN
        Tax Rate For Calcs                                            0.11464  ...          NaN
        Normalized EBITDA                                        4172000000.0  ...          NaN
        """
    try:
        dat = get_ticker_data(stock_ticker)
        return str(f"Income statement for {stock_ticker}: {dat.quarterly_income_stmt}")
    except Exception as e:
        logger.error(f"Error retrieving income statement for {stock_ticker}: {e}")
        time.sleep(1)  # Wait before retrying
        try:
            dat = get_ticker_data(stock_ticker)
            return str(f"Income statement for {stock_ticker}: {dat.quarterly_income_stmt}")
        except Exception as e2:
            return f"Error retrieving income statement for {stock_ticker}: {str(e2)}"

# Kick off server if file is run 
if __name__ == "__main__":
    mcp.run(transport="stdio")