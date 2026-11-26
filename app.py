import pandas as pd

containers = pd.read_csv("AtmoSync_Synthetic_Dataset/containers.csv")
readings = pd.read_csv("AtmoSync_Synthetic_Dataset/sensor_readings.csv")
prices = pd.read_csv("AtmoSync_Synthetic_Dataset/commodity_market_prices.csv")
routes = pd.read_csv("AtmoSync_Synthetic_Dataset/market_routes.csv")
