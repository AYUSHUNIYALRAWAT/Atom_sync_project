-- Run this script after connecting to the bookstore_sales database.
-- Table names are lowercase for consistent PostgreSQL usage.

CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    author VARCHAR(150),
    genre VARCHAR(100),
    published_year INTEGER,
    price NUMERIC(10, 2),
    stock INTEGER
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    name VARCHAR(150) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(30),
    city VARCHAR(150),
    country VARCHAR(150)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    book_id INTEGER REFERENCES books(book_id),
    order_date DATE,
    quantity INTEGER CHECK (quantity >= 0),
    total_amount NUMERIC(12, 2)
);
