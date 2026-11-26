# Bookstore Sales Analysis Using SQL

A SQL portfolio project exploring bookstore inventory, book genres, customer locations, order activity, and sales performance.

## Project objectives

- Explore books by genre, publication year, price, and stock.
- Analyze customer locations and repeat purchasing.
- Review order dates, quantities, and recorded revenue.
- Identify best-selling books and authors.
- Summarize sales by genre and customer city.

## Tech stack

- PostgreSQL
- SQL
- CSV datasets
- Git and GitHub

## Repository structure

```text
bookstore-sales-sql-analysis/
├── README.md
├── 01_create_database.sql
├── 02_create_tables.sql
├── 03_load_data.sql
├── 04_basic_analysis.sql
├── 05_advanced_analysis.sql
├── data/
│   ├── Books.csv
│   ├── Customers.csv
│   └── Orders.csv
└── results/
    └── (optional screenshots or exported results)
```

## Dataset

The supplied CSV files contain:

- `Books.csv`: `Book_ID`, `Title`, `Author`, `Genre`, `Published_Year`, `Price`, `Stock`
- `Customers.csv`: `Customer_ID`, `Name`, `Email`, `Phone`, `City`, `Country`
- `Orders.csv`: `Order_ID`, `Customer_ID`, `Book_ID`, `Order_Date`, `Quantity`, `Total_Amount`

Each CSV currently contains 500 rows.

## How to run (PostgreSQL)

### Option A: Using psql

Run commands from the repository root. First create the database:

```bash
psql -d postgres -f 01_create_database.sql
```

Then create tables and load the CSV files:

```bash
psql -d bookstore_sales -f 02_create_tables.sql
psql -d bookstore_sales -f 03_load_data.sql
```

Run the analysis scripts:

```bash
psql -d bookstore_sales -f 04_basic_analysis.sql
psql -d bookstore_sales -f 05_advanced_analysis.sql
```

The `03_load_data.sql` script uses PostgreSQL's `\\copy` command and expects the terminal's working directory to be the repository root. Run it only once against empty tables to avoid duplicate primary-key errors.

### Option B: Using pgAdmin

1. Connect to PostgreSQL.
2. Open the Query Tool against an existing database and execute `01_create_database.sql` separately.
3. Connect to the new `bookstore_sales` database.
4. Execute `02_create_tables.sql`.
5. For each table, use **Import/Export Data** to import the matching CSV from the `data/` folder, with Header enabled and the correct columns mapped.
6. Execute `04_basic_analysis.sql` and `05_advanced_analysis.sql`.

## SQL concepts demonstrated

- `SELECT`, `WHERE`, `DISTINCT`
- Date filtering
- `ORDER BY` and `LIMIT`
- Aggregate functions: `SUM`, `AVG`, `COUNT`
- `GROUP BY` and `HAVING`
- `JOIN`
- Business analysis and KPI summaries

## Important interpretation notes

- “Most-sold book” is calculated using `SUM(quantity)`, rather than counting order records.
- “More than two orders” uses `HAVING COUNT(order_id) > 2`.
- City spending is calculated as the sum of order amounts per city.
- `total_amount` is treated as the revenue amount already recorded in the orders dataset; the scripts do not assume it should be recalculated from price and quantity.

## Results and insights

Add findings only after running the queries, for example:
- Top-selling book by units sold
- Highest-revenue genre
- Total recorded revenue
- Most common customer locations
- Books with the lowest inventory

## Author

Add your name and GitHub profile here.
