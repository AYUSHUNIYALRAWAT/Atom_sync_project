-- ADVANCED ANALYSIS QUERIES
-- Bookstore Sales Analysis | PostgreSQL

-- 1. Find the total quantity of books sold in each genre
SELECT
    b.genre,
    SUM(o.quantity) AS total_books_sold
FROM orders AS o
JOIN books AS b ON b.book_id = o.book_id
GROUP BY b.genre
ORDER BY total_books_sold DESC;

-- 2. Find the average price of books in the Fantasy genre
SELECT AVG(price) AS average_fantasy_book_price
FROM books
WHERE genre = 'Fantasy';

-- 3. List customers who placed more than two orders
SELECT
    c.customer_id,
    c.name,
    COUNT(o.order_id) AS order_count
FROM customers AS c
JOIN orders AS o ON o.customer_id = c.customer_id
GROUP BY c.customer_id, c.name
HAVING COUNT(o.order_id) > 2
ORDER BY order_count DESC;

-- 4. Find the most-sold book by quantity
SELECT
    b.book_id,
    b.title,
    SUM(o.quantity) AS total_quantity_sold
FROM orders AS o
JOIN books AS b ON b.book_id = o.book_id
GROUP BY b.book_id, b.title
ORDER BY total_quantity_sold DESC
LIMIT 1;

-- 5. Find the top three most expensive Fantasy books
SELECT *
FROM books
WHERE genre = 'Fantasy'
ORDER BY price DESC
LIMIT 3;

-- 6. Retrieve the total quantity of books sold by each author
SELECT
    b.author,
    SUM(o.quantity) AS total_books_sold
FROM books AS b
JOIN orders AS o ON o.book_id = b.book_id
GROUP BY b.author
ORDER BY total_books_sold DESC;

-- 7. List cities where combined order spending exceeds $30
SELECT
    c.city,
    SUM(o.total_amount) AS total_spending
FROM customers AS c
JOIN orders AS o ON o.customer_id = c.customer_id
GROUP BY c.city
HAVING SUM(o.total_amount) > 30
ORDER BY total_spending DESC;

-- 8. Optional: Count order records and total units sold by genre
SELECT
    b.genre,
    COUNT(DISTINCT o.order_id) AS order_count,
    SUM(o.quantity) AS units_sold,
    SUM(o.total_amount) AS recorded_revenue
FROM books AS b
JOIN orders AS o ON o.book_id = b.book_id
GROUP BY b.genre
ORDER BY recorded_revenue DESC;
