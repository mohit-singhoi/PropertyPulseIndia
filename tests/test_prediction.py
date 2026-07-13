import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction import predict_price, predict_rent, get_city_avg_prices

class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.test_input = {
            'city': 'Delhi',
            'property_type': 'Apartment',
            'area': 1200,
            'bedrooms': 3,
            'bathrooms': 2,
            'balcony': 1,
            'hall': 1,
            'furnishing': 'Furnished',
            'property_age': 'New'
        }
    
    def test_predict_price(self):
        """Test price prediction"""
        try:
            price, confidence = predict_price(self.test_input)
            
            self.assertIsNotNone(price)
            self.assertIsNotNone(confidence)
            self.assertGreater(price, 0)
            self.assertEqual(len(confidence), 2)
            self.assertLess(confidence[0], confidence[1])
            print(f"✅ Price prediction test passed: ₹{price:,.2f}")
        except Exception as e:
            self.fail(f"predict_price failed: {e}")
    
    def test_predict_rent(self):
        """Test rent prediction"""
        try:
            rent, confidence = predict_rent(self.test_input)
            
            self.assertIsNotNone(rent)
            self.assertIsNotNone(confidence)
            self.assertGreater(rent, 0)
            self.assertEqual(len(confidence), 2)
            self.assertLess(confidence[0], confidence[1])
            print(f"✅ Rent prediction test passed: ₹{rent:,.2f}")
        except Exception as e:
            self.fail(f"predict_rent failed: {e}")
    
    def test_get_city_avg_prices(self):
        """Test city averages"""
        avg = get_city_avg_prices('Delhi')
        
        self.assertIn('sale', avg)
        self.assertIn('rent', avg)
        self.assertGreater(avg['sale'], 0)
        self.assertGreater(avg['rent'], 0)
        print(f"✅ City avg prices test passed: Sale ₹{avg['sale']:,.2f}, Rent ₹{avg['rent']:,.2f}")
    
    def test_invalid_city(self):
        """Test invalid city handling"""
        avg = get_city_avg_prices('InvalidCity')
        
        self.assertIn('sale', avg)
        self.assertIn('rent', avg)
        self.assertGreater(avg['sale'], 0)
        self.assertGreater(avg['rent'], 0)
        print(f"✅ Invalid city test passed: {avg}")

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)