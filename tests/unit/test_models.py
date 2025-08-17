import unittest
import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.competitor import Competitor
from models.product import Product
from models.sales_result import SalesResult


class TestCompetitor(unittest.TestCase):
    """Test suite for the Competitor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.competitor_data = {
            "id": "competitor1",
            "name": "Competitor A",
            "market_share": 0.25,
            "strengths": ["brand_recognition", "distribution_network"],
            "weaknesses": ["high_prices", "limited_product_range"],
            "products": ["product1", "product2"],
            "pricing_strategy": "premium",
            "market_position": "leader",
            "recent_activities": ["launched_new_product", "expanded_to_new_market"]
        }
        self.competitor = Competitor(self.competitor_data)
    
    def test_competitor_initialization(self):
        """Test that the competitor initializes correctly."""
        self.assertIsNotNone(self.competitor)
        self.assertEqual(self.competitor.id, "competitor1")
        self.assertEqual(self.competitor.name, "Competitor A")
        self.assertEqual(self.competitor.market_share, 0.25)
        self.assertEqual(self.competitor.strengths, ["brand_recognition", "distribution_network"])
        self.assertEqual(self.competitor.weaknesses, ["high_prices", "limited_product_range"])
        self.assertEqual(self.competitor.products, ["product1", "product2"])
        self.assertEqual(self.competitor.pricing_strategy, "premium")
        self.assertEqual(self.competitor.market_position, "leader")
        self.assertEqual(self.competitor.recent_activities, ["launched_new_product", "expanded_to_new_market"])
    
    def test_competitor_to_dict(self):
        """Test converting competitor to dictionary."""
        competitor_dict = self.competitor.to_dict()
        
        self.assertIsInstance(competitor_dict, dict)
        self.assertEqual(competitor_dict["id"], "competitor1")
        self.assertEqual(competitor_dict["name"], "Competitor A")
        self.assertEqual(competitor_dict["market_share"], 0.25)
        self.assertEqual(competitor_dict["strengths"], ["brand_recognition", "distribution_network"])
        self.assertEqual(competitor_dict["weaknesses"], ["high_prices", "limited_product_range"])
        self.assertEqual(competitor_dict["products"], ["product1", "product2"])
        self.assertEqual(competitor_dict["pricing_strategy"], "premium")
        self.assertEqual(competitor_dict["market_position"], "leader")
        self.assertEqual(competitor_dict["recent_activities"], ["launched_new_product", "expanded_to_new_market"])
    
    def test_competitor_from_dict(self):
        """Test creating competitor from dictionary."""
        competitor_dict = {
            "id": "competitor2",
            "name": "Competitor B",
            "market_share": 0.15,
            "strengths": ["low_prices", "wide_product_range"],
            "weaknesses": ["weak_brand", "poor_customer_service"],
            "products": ["product3", "product4"],
            "pricing_strategy": "economy",
            "market_position": "follower",
            "recent_activities": ["price_cut", "product_discontinuation"]
        }
        
        competitor = Competitor.from_dict(competitor_dict)
        
        self.assertIsNotNone(competitor)
        self.assertEqual(competitor.id, "competitor2")
        self.assertEqual(competitor.name, "Competitor B")
        self.assertEqual(competitor.market_share, 0.15)
        self.assertEqual(competitor.strengths, ["low_prices", "wide_product_range"])
        self.assertEqual(competitor.weaknesses, ["weak_brand", "poor_customer_service"])
        self.assertEqual(competitor.products, ["product3", "product4"])
        self.assertEqual(competitor.pricing_strategy, "economy")
        self.assertEqual(competitor.market_position, "follower")
        self.assertEqual(competitor.recent_activities, ["price_cut", "product_discontinuation"])
    
    def test_competitor_update_market_share(self):
        """Test updating competitor market share."""
        self.competitor.update_market_share(0.30)
        
        self.assertEqual(self.competitor.market_share, 0.30)
    
    def test_competitor_add_strength(self):
        """Test adding a strength to competitor."""
        self.competitor.add_strength("customer_loyalty")
        
        self.assertIn("customer_loyalty", self.competitor.strengths)
    
    def test_competitor_add_weakness(self):
        """Test adding a weakness to competitor."""
        self.competitor.add_weakness("supply_chain_issues")
        
        self.assertIn("supply_chain_issues", self.competitor.weaknesses)
    
    def test_competitor_add_product(self):
        """Test adding a product to competitor."""
        self.competitor.add_product("product3")
        
        self.assertIn("product3", self.competitor.products)
    
    def test_competitor_add_recent_activity(self):
        """Test adding a recent activity to competitor."""
        self.competitor.add_recent_activity("acquired_company")
        
        self.assertIn("acquired_company", self.competitor.recent_activities)
    
    def test_competitor_calculate_threat_level(self):
        """Test calculating competitor threat level."""
        threat_level = self.competitor.calculate_threat_level()
        
        self.assertIsNotNone(threat_level)
        self.assertGreaterEqual(threat_level, 0.0)
        self.assertLessEqual(threat_level, 1.0)
    
    def test_competitor_get_competitive_advantage(self):
        """Test getting competitor competitive advantage."""
        advantage = self.competitor.get_competitive_advantage()
        
        self.assertIsNotNone(advantage)
        self.assertIsInstance(advantage, list)
        self.assertGreater(len(advantage), 0)


class TestProduct(unittest.TestCase):
    """Test suite for the Product class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.product_data = {
            "id": "product1",
            "name": "Product A",
            "category": "electronics",
            "price": 99.99,
            "cost": 50.0,
            "description": "A high-quality electronic product",
            "features": ["feature1", "feature2", "feature3"],
            "specifications": {"weight": "1kg", "dimensions": "10x20x30cm"},
            "inventory": 100,
            "sales_history": [
                {"date": "2023-01-01", "quantity": 10, "revenue": 999.9},
                {"date": "2023-01-02", "quantity": 15, "revenue": 1499.85}
            ],
            "customer_reviews": [
                {"rating": 5, "comment": "Great product!"},
                {"rating": 4, "comment": "Good value for money."}
            ],
            "competitors": ["competitor1", "competitor2"]
        }
        self.product = Product(self.product_data)
    
    def test_product_initialization(self):
        """Test that the product initializes correctly."""
        self.assertIsNotNone(self.product)
        self.assertEqual(self.product.id, "product1")
        self.assertEqual(self.product.name, "Product A")
        self.assertEqual(self.product.category, "electronics")
        self.assertEqual(self.product.price, 99.99)
        self.assertEqual(self.product.cost, 50.0)
        self.assertEqual(self.product.description, "A high-quality electronic product")
        self.assertEqual(self.product.features, ["feature1", "feature2", "feature3"])
        self.assertEqual(self.product.specifications, {"weight": "1kg", "dimensions": "10x20x30cm"})
        self.assertEqual(self.product.inventory, 100)
        self.assertEqual(len(self.product.sales_history), 2)
        self.assertEqual(len(self.product.customer_reviews), 2)
        self.assertEqual(self.product.competitors, ["competitor1", "competitor2"])
    
    def test_product_to_dict(self):
        """Test converting product to dictionary."""
        product_dict = self.product.to_dict()
        
        self.assertIsInstance(product_dict, dict)
        self.assertEqual(product_dict["id"], "product1")
        self.assertEqual(product_dict["name"], "Product A")
        self.assertEqual(product_dict["category"], "electronics")
        self.assertEqual(product_dict["price"], 99.99)
        self.assertEqual(product_dict["cost"], 50.0)
        self.assertEqual(product_dict["description"], "A high-quality electronic product")
        self.assertEqual(product_dict["features"], ["feature1", "feature2", "feature3"])
        self.assertEqual(product_dict["specifications"], {"weight": "1kg", "dimensions": "10x20x30cm"})
        self.assertEqual(product_dict["inventory"], 100)
        self.assertEqual(len(product_dict["sales_history"]), 2)
        self.assertEqual(len(product_dict["customer_reviews"]), 2)
        self.assertEqual(product_dict["competitors"], ["competitor1", "competitor2"])
    
    def test_product_from_dict(self):
        """Test creating product from dictionary."""
        product_dict = {
            "id": "product2",
            "name": "Product B",
            "category": "clothing",
            "price": 49.99,
            "cost": 25.0,
            "description": "A comfortable clothing item",
            "features": ["feature4", "feature5"],
            "specifications": {"size": "M", "material": "cotton"},
            "inventory": 200,
            "sales_history": [
                {"date": "2023-01-01", "quantity": 20, "revenue": 999.8}
            ],
            "customer_reviews": [
                {"rating": 3, "comment": "Average product."}
            ],
            "competitors": ["competitor3", "competitor4"]
        }
        
        product = Product.from_dict(product_dict)
        
        self.assertIsNotNone(product)
        self.assertEqual(product.id, "product2")
        self.assertEqual(product.name, "Product B")
        self.assertEqual(product.category, "clothing")
        self.assertEqual(product.price, 49.99)
        self.assertEqual(product.cost, 25.0)
        self.assertEqual(product.description, "A comfortable clothing item")
        self.assertEqual(product.features, ["feature4", "feature5"])
        self.assertEqual(product.specifications, {"size": "M", "material": "cotton"})
        self.assertEqual(product.inventory, 200)
        self.assertEqual(len(product.sales_history), 1)
        self.assertEqual(len(product.customer_reviews), 1)
        self.assertEqual(product.competitors, ["competitor3", "competitor4"])
    
    def test_product_update_price(self):
        """Test updating product price."""
        self.product.update_price(89.99)
        
        self.assertEqual(self.product.price, 89.99)
    
    def test_product_update_inventory(self):
        """Test updating product inventory."""
        self.product.update_inventory(150)
        
        self.assertEqual(self.product.inventory, 150)
    
    def test_product_add_feature(self):
        """Test adding a feature to product."""
        self.product.add_feature("feature4")
        
        self.assertIn("feature4", self.product.features)
    
    def test_product_add_sale(self):
        """Test adding a sale to product sales history."""
        sale_data = {
            "date": "2023-01-03",
            "quantity": 20,
            "revenue": 1999.8
        }
        
        self.product.add_sale(sale_data)
        
        self.assertEqual(len(self.product.sales_history), 3)
        self.assertEqual(self.product.sales_history[2]["quantity"], 20)
        self.assertEqual(self.product.sales_history[2]["revenue"], 1999.8)
    
    def test_product_add_customer_review(self):
        """Test adding a customer review to product."""
        review_data = {
            "rating": 3,
            "comment": "Average product."
        }
        
        self.product.add_customer_review(review_data)
        
        self.assertEqual(len(self.product.customer_reviews), 3)
        self.assertEqual(self.product.customer_reviews[2]["rating"], 3)
        self.assertEqual(self.product.customer_reviews[2]["comment"], "Average product.")
    
    def test_product_add_competitor(self):
        """Test adding a competitor to product."""
        self.product.add_competitor("competitor3")
        
        self.assertIn("competitor3", self.product.competitors)
    
    def test_product_calculate_profit_margin(self):
        """Test calculating product profit margin."""
        profit_margin = self.product.calculate_profit_margin()
        
        self.assertIsNotNone(profit_margin)
        self.assertEqual(profit_margin, 0.5)  # (99.99 - 50.0) / 99.99 (approximately)
    
    def test_product_calculate_average_rating(self):
        """Test calculating product average rating."""
        average_rating = self.product.calculate_average_rating()
        
        self.assertIsNotNone(average_rating)
        self.assertEqual(average_rating, 4.5)  # (5 + 4) / 2
    
    def test_product_calculate_total_sales(self):
        """Test calculating product total sales."""
        total_sales = self.product.calculate_total_sales()
        
        self.assertIsNotNone(total_sales)
        self.assertEqual(total_sales, 25)  # 10 + 15
    
    def test_product_calculate_total_revenue(self):
        """Test calculating product total revenue."""
        total_revenue = self.product.calculate_total_revenue()
        
        self.assertIsNotNone(total_revenue)
        self.assertEqual(total_revenue, 2499.75)  # 999.9 + 1499.85
    
    def test_product_get_best_selling_period(self):
        """Test getting product best selling period."""
        best_period = self.product.get_best_selling_period()
        
        self.assertIsNotNone(best_period)
        self.assertEqual(best_period["date"], "2023-01-02")
        self.assertEqual(best_period["quantity"], 15)


class TestSalesResult(unittest.TestCase):
    """Test suite for the SalesResult class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sales_result_data = {
            "id": "sales1",
            "product_id": "product1",
            "sales_agent_id": "agent1",
            "customer_id": "customer1",
            "quantity": 10,
            "unit_price": 99.99,
            "total_price": 999.9,
            "discount": 0.0,
            "sale_date": "2023-01-01",
            "sale_channel": "online",
            "payment_method": "credit_card",
            "shipping_address": "123 Main St, City, Country",
            "order_status": "completed",
            "delivery_date": "2023-01-05",
            "customer_feedback": "Great service!",
            "commission_rate": 0.1,
            "commission_amount": 99.99
        }
        self.sales_result = SalesResult(self.sales_result_data)
    
    def test_sales_result_initialization(self):
        """Test that the sales result initializes correctly."""
        self.assertIsNotNone(self.sales_result)
        self.assertEqual(self.sales_result.id, "sales1")
        self.assertEqual(self.sales_result.product_id, "product1")
        self.assertEqual(self.sales_result.sales_agent_id, "agent1")
        self.assertEqual(self.sales_result.customer_id, "customer1")
        self.assertEqual(self.sales_result.quantity, 10)
        self.assertEqual(self.sales_result.unit_price, 99.99)
        self.assertEqual(self.sales_result.total_price, 999.9)
        self.assertEqual(self.sales_result.discount, 0.0)
        self.assertEqual(self.sales_result.sale_date, "2023-01-01")
        self.assertEqual(self.sales_result.sale_channel, "online")
        self.assertEqual(self.sales_result.payment_method, "credit_card")
        self.assertEqual(self.sales_result.shipping_address, "123 Main St, City, Country")
        self.assertEqual(self.sales_result.order_status, "completed")
        self.assertEqual(self.sales_result.delivery_date, "2023-01-05")
        self.assertEqual(self.sales_result.customer_feedback, "Great service!")
        self.assertEqual(self.sales_result.commission_rate, 0.1)
        self.assertEqual(self.sales_result.commission_amount, 99.99)
    
    def test_sales_result_to_dict(self):
        """Test converting sales result to dictionary."""
        sales_result_dict = self.sales_result.to_dict()
        
        self.assertIsInstance(sales_result_dict, dict)
        self.assertEqual(sales_result_dict["id"], "sales1")
        self.assertEqual(sales_result_dict["product_id"], "product1")
        self.assertEqual(sales_result_dict["sales_agent_id"], "agent1")
        self.assertEqual(sales_result_dict["customer_id"], "customer1")
        self.assertEqual(sales_result_dict["quantity"], 10)
        self.assertEqual(sales_result_dict["unit_price"], 99.99)
        self.assertEqual(sales_result_dict["total_price"], 999.9)
        self.assertEqual(sales_result_dict["discount"], 0.0)
        self.assertEqual(sales_result_dict["sale_date"], "2023-01-01")
        self.assertEqual(sales_result_dict["sale_channel"], "online")
        self.assertEqual(sales_result_dict["payment_method"], "credit_card")
        self.assertEqual(sales_result_dict["shipping_address"], "123 Main St, City, Country")
        self.assertEqual(sales_result_dict["order_status"], "completed")
        self.assertEqual(sales_result_dict["delivery_date"], "2023-01-05")
        self.assertEqual(sales_result_dict["customer_feedback"], "Great service!")
        self.assertEqual(sales_result_dict["commission_rate"], 0.1)
        self.assertEqual(sales_result_dict["commission_amount"], 99.99)
    
    def test_sales_result_from_dict(self):
        """Test creating sales result from dictionary."""
        sales_result_dict = {
            "id": "sales2",
            "product_id": "product2",
            "sales_agent_id": "agent2",
            "customer_id": "customer2",
            "quantity": 5,
            "unit_price": 49.99,
            "total_price": 249.95,
            "discount": 10.0,
            "sale_date": "2023-01-02",
            "sale_channel": "retail",
            "payment_method": "cash",
            "shipping_address": "456 Oak St, City, Country",
            "order_status": "pending",
            "delivery_date": "2023-01-07",
            "customer_feedback": "Good product.",
            "commission_rate": 0.05,
            "commission_amount": 12.5
        }
        
        sales_result = SalesResult.from_dict(sales_result_dict)
        
        self.assertIsNotNone(sales_result)
        self.assertEqual(sales_result.id, "sales2")
        self.assertEqual(sales_result.product_id, "product2")
        self.assertEqual(sales_result.sales_agent_id, "agent2")
        self.assertEqual(sales_result.customer_id, "customer2")
        self.assertEqual(sales_result.quantity, 5)
        self.assertEqual(sales_result.unit_price, 49.99)
        self.assertEqual(sales_result.total_price, 249.95)
        self.assertEqual(sales_result.discount, 10.0)
        self.assertEqual(sales_result.sale_date, "2023-01-02")
        self.assertEqual(sales_result.sale_channel, "retail")
        self.assertEqual(sales_result.payment_method, "cash")
        self.assertEqual(sales_result.shipping_address, "456 Oak St, City, Country")
        self.assertEqual(sales_result.order_status, "pending")
        self.assertEqual(sales_result.delivery_date, "2023-01-07")
        self.assertEqual(sales_result.customer_feedback, "Good product.")
        self.assertEqual(sales_result.commission_rate, 0.05)
        self.assertEqual(sales_result.commission_amount, 12.5)
    
    def test_sales_result_update_quantity(self):
        """Test updating sales result quantity."""
        self.sales_result.update_quantity(15)
        
        self.assertEqual(self.sales_result.quantity, 15)
        self.assertEqual(self.sales_result.total_price, 1499.85)  # 15 * 99.99
    
    def test_sales_result_update_unit_price(self):
        """Test updating sales result unit price."""
        self.sales_result.update_unit_price(89.99)
        
        self.assertEqual(self.sales_result.unit_price, 89.99)
        self.assertEqual(self.sales_result.total_price, 899.9)  # 10 * 89.99
    
    def test_sales_result_apply_discount(self):
        """Test applying discount to sales result."""
        self.sales_result.apply_discount(10.0)  # 10% discount
        
        self.assertEqual(self.sales_result.discount, 10.0)
        self.assertEqual(self.sales_result.total_price, 899.91)  # 999.9 * 0.9
    
    def test_sales_result_update_order_status(self):
        """Test updating sales result order status."""
        self.sales_result.update_order_status("shipped")
        
        self.assertEqual(self.sales_result.order_status, "shipped")
    
    def test_sales_result_update_delivery_date(self):
        """Test updating sales result delivery date."""
        self.sales_result.update_delivery_date("2023-01-06")
        
        self.assertEqual(self.sales_result.delivery_date, "2023-01-06")
    
    def test_sales_result_add_customer_feedback(self):
        """Test adding customer feedback to sales result."""
        self.sales_result.add_customer_feedback("Excellent product!")
        
        self.assertEqual(self.sales_result.customer_feedback, "Excellent product!")
    
    def test_sales_result_calculate_commission(self):
        """Test calculating sales result commission."""
        commission = self.sales_result.calculate_commission()
        
        self.assertIsNotNone(commission)
        self.assertEqual(commission, 99.99)  # 999.9 * 0.1
    
    def test_sales_result_calculate_profit(self):
        """Test calculating sales result profit."""
        profit = self.sales_result.calculate_profit(50.0)  # cost per unit
        
        self.assertIsNotNone(profit)
        self.assertEqual(profit, 499.9)  # (99.99 - 50.0) * 10
    
    def test_sales_result_calculate_delivery_time(self):
        """Test calculating sales result delivery time."""
        delivery_time = self.sales_result.calculate_delivery_time()
        
        self.assertIsNotNone(delivery_time)
        self.assertEqual(delivery_time, 4)  # 2023-01-05 - 2023-01-01 = 4 days


if __name__ == '__main__':
    unittest.main()