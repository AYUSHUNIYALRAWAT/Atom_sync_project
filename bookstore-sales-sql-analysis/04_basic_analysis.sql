-- BASIC ANALYSIS QUERIES
-- Bookstore Sales Analysis | PostgreSQL

-- 1. Retrieve all books in the Fiction genre
SELECT *
FROM books
WHERE genre = 'Fiction';

-- 2. Find books published after 1950
SELECT *
FROM books
WHERE published_year > 1950;

-- 3. List all customers from Canada
SELECT *
FROM customers
WHERE country = 'Canada';

-- 4. Show all orders placed in November 2023
SELECT *
FROM orders
WHERE order_date >= DATE '2023-11-01'
  AND order_date <  DATE '2023-12-01';

-- 5. Retrieve the total stock of books available
SELECT SUM(stock) AS total_stock
FROM books;

-- 6. Find the most expensive book
SELECT *
FROM books
ORDER BY price DESC
LIMIT 1;

-- 7. Show orders containing more than one book
SELECT *
FROM orders
WHERE quantity > 1;

-- 8. Retrieve orders where the total amount exceeds $20
SELECT *
FROM orders
WHERE total_amount > 20;

-- 9. List all genres available in the books table
SELECT DISTINCT genre
FROM books
ORDER BY genre;

-- 10. Find the book with the lowest stock
SELECT *
FROM books
ORDER BY stock ASC
LIMIT 1;

-- 11. Calculate total revenue recorded in orders
SELECT SUM(total_amount) AS total_revenue
FROM orders;
