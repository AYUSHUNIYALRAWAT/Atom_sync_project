AtmoSync synthetic project dataset
=====================================

Files
- containers.csv: one row per shipment/container and commodity-specific illustrative storage targets.
- sensor_readings.csv: simulated sensor readings every 15 minutes for 100 containers over 7 days.
- commodity_market_prices.csv: synthetic daily commodity prices, demand index and market capacity.
- market_routes.csv: approximate synthetic route distances, travel-time estimates and extra transport costs.

Important
- This is synthetic/mock data generated for project development and demonstration. It is NOT real sensor,
  market-price, route, or food-safety data.
- Temperature/humidity targets and risk labels are illustrative only. Do not use them for real food-safety,
  logistics, or commercial decisions. Validate thresholds with authoritative commodity-specific standards.
- The risk_label is a simple rule-based example, not a scientifically validated spoilage prediction.
- Currency is INR. Distances are approximate.

Suggested joins
- sensor_readings.container_id -> containers.container_id
- commodity_market_prices on commodity and date (date derived from sensor timestamp)
- market_routes on origin_market + destination_market

Suggested project tasks
1. Clean missing sensor records and check duplicates/outliers.
2. Build temperature/humidity time-series charts per container.
3. Calculate high/medium/low rule-based risk counts.
4. Compare destination market prices and estimate net rerouting value:
   (alternative price - original price) * quantity - extra_transport_cost.
5. Build a dashboard for container health, market comparison and potential rerouting.
