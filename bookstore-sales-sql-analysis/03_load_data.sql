-- PostgreSQL psql script.
-- Run from the project root after connecting to the bookstore_sales database:
--   psql -d bookstore_sales -f 03_load_data.sql
-- If using pgAdmin, you can instead import each CSV through the table's Import/Export Data menu.
-- The tables must already exist and be empty before loading.

\copy books(book_id, title, author, genre, published_year, price, stock) FROM 'data/Books.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');
\copy customers(customer_id, name, email, phone, city, country) FROM 'data/Customers.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');
\copy orders(order_id, customer_id, book_id, order_date, quantity, total_amount) FROM 'data/Orders.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');
