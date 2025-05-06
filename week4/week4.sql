
SELECT COUNT(DISTINCT order_id) AS total_orders
FROM SALES
WHERE DATE = '2023-03-18';

SELECT COUNT(DISTINCT S.order_id) AS total_orders
FROM SALES S
JOIN CUSTOMERS C ON S.customer_id = C.customer_id
WHERE S.DATE = '2023-03-18'
  AND (C.first_name = 'John' OR C.last_name = 'Doe');


SELECT 
  COUNT(DISTINCT customer_id) AS total_customers,
  ROUND(SUM(revenue) / COUNT(DISTINCT customer_id), 2) AS avg_spent_per_customer
FROM SALES
WHERE DATE BETWEEN '2023-01-01' AND '2023-12-31';


SELECT I.department, SUM(S.revenue) AS total_revenue
FROM SALES S
JOIN ITEMS I ON S.item_id = I.item_id
WHERE S.DATE BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY I.department
HAVING SUM(S.revenue) < 800;


SELECT order_id, SUM(revenue) AS order_revenue
FROM SALES
GROUP BY order_id
ORDER BY order_revenue DESC;


WITH OrderTotals AS (
  SELECT order_id, SUM(revenue) AS total_revenue
  FROM SALES
  GROUP BY order_id
  ORDER BY total_revenue DESC
  LIMIT 1
)
SELECT S.order_id, I.item_name, S.quantity, S.revenue
FROM SALES S
JOIN ITEMS I ON S.item_id = I.item_id
JOIN OrderTotals OT ON S.order_id = OT.order_id;
