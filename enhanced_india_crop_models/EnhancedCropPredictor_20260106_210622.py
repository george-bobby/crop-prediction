
import joblib
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Optional, Union

class EnhancedIndiaCropPredictor:
    """
    Enhanced production-ready predictor for Indian crop yields.
    Uses soil nutrients, climate data, and agricultural features.
    """

    def __init__(self, model_path: str, preprocessing_path: str):
        """
        Initialize enhanced predictor.

        Args:
            model_path: Path to trained model pickle file
            preprocessing_path: Path to preprocessing objects pickle file
        """
        self.model = joblib.load(model_path)

        with open(preprocessing_path, 'rb') as f:
            self.preprocessing = pickle.load(f)

        # Extract preprocessing components
        self.feature_names = self.preprocessing['feature_names']
        self.label_encoders = self.preprocessing.get('label_encoders', {})
        self.crop_encoder = self.preprocessing.get('crop_encoder', None)
        self.state_encoder = self.preprocessing.get('state_encoder', None)

        # Feature categories
        self.soil_features = self.preprocessing.get('soil_features', [])
        self.climate_features = self.preprocessing.get('climate_features', [])
        self.production_features = self.preprocessing.get('production_features', [])
        self.encoded_features = self.preprocessing.get('encoded_features', [])

        print(f"✅ EnhancedIndiaCropPredictor loaded successfully")
        print(f"   Model: {self.preprocessing['dataset_info']['best_model']}")
        print(f"   Performance: R²={self.preprocessing['dataset_info']['best_r2']:.4f}")
        print(f"   Features: {len(self.feature_names)} total")

    def _calculate_derived_features(self, input_data: Dict) -> Dict:
        """
        Calculate derived features from input data.
        """
        features = input_data.copy()

        # Calculate nutrient ratios
        if 'N' in features and 'P' in features:
            features['N_P_ratio'] = features['N'] / (features['P'] + 1)
        if 'N' in features and 'K' in features:
            features['N_K_ratio'] = features['N'] / (features['K'] + 1)
        if 'P' in features and 'K' in features:
            features['P_K_ratio'] = features['P'] / (features['K'] + 1)
        if all(x in features for x in ['N', 'P', 'K']):
            features['total_nutrients'] = features['N'] + features['P'] + features['K']

        # Calculate pH category
        if 'pH' in features:
            ph = features['pH']
            if ph < 5.0:
                features['pH_category'] = 'Very Acidic'
            elif ph < 5.5:
                features['pH_category'] = 'Acidic'
            elif ph < 6.5:
                features['pH_category'] = 'Slightly Acidic'
            elif ph < 7.0:
                features['pH_category'] = 'Neutral'
            elif ph < 7.5:
                features['pH_category'] = 'Slightly Alkaline'
            else:
                features['pH_category'] = 'Alkaline'

            features['pH_optimal'] = 1 if 5.5 <= ph <= 7.0 else 0

        # Calculate temperature suitability
        if all(x in features for x in ['Crop', 'temperature']):
            crop = features['Crop'].lower()
            temp = features['temperature']

            optimal_temp_ranges = {
                'rice': (20, 35), 'wheat': (10, 25), 'maize': (18, 32),
                'cotton': (20, 30), 'sugarcane': (20, 35), 'groundnut': (20, 30)
            }

            suitability = 0.5  # Default
            for key, (min_temp, max_temp) in optimal_temp_ranges.items():
                if key in crop:
                    if min_temp <= temp <= max_temp:
                        suitability = 1.0
                    elif temp < min_temp - 5 or temp > max_temp + 5:
                        suitability = 0.3
                    else:
                        suitability = 0.7
                    break

            features['temp_suitability'] = suitability

        # Calculate rainfall adequacy
        if 'rainfall' in features:
            rainfall = features['rainfall']
            if rainfall > 500:
                features['rainfall_adequacy'] = 1
            elif rainfall > 300:
                features['rainfall_adequacy'] = 0.7
            else:
                features['rainfall_adequacy'] = 0.4

        # Determine crop category
        if 'Crop' in features:
            crop_name = features['Crop'].lower()
            crop_categories = {
                'cereals': ['rice', 'wheat', 'maize', 'jowar', 'bajra', 'ragi'],
                'pulses': ['gram', 'pigeonpeas', 'moong', 'urd', 'lentil', 'peas'],
                'oilseeds': ['groundnut', 'mustard', 'soybean', 'sunflower', 'sesamum'],
                'cash crops': ['sugarcane', 'cotton', 'tobacco', 'jute'],
                'vegetables': ['potato', 'onion', 'tomato', 'brinjal', 'cabbage', 'cauliflower']
            }

            crop_category = 'others'
            for category, crops in crop_categories.items():
                for crop in crops:
                    if crop in crop_name:
                        crop_category = category
                        break
                if crop_category != 'others':
                    break

            features['crop_category'] = crop_category

        # Add state yield statistics (use defaults if not available)
        if 'State' in features:
            # In production, you would load actual state statistics
            # For now, use reasonable defaults
            features['state_yield_mean'] = 2.0  # Average yield in tons/ha
            features['state_yield_std'] = 1.0   # Standard deviation

        return features

    def _encode_features(self, features: Dict) -> pd.DataFrame:
        """
        Encode categorical features.
        """
        encoded = features.copy()

        # Apply target encoding for Crop
        if self.crop_encoder is not None and 'Crop' in encoded:
            try:
                encoded['Crop_Encoded'] = self.crop_encoder.transform(
                    pd.DataFrame({'Crop': [encoded['Crop']]})
                ).iloc[0, 0]
            except:
                encoded['Crop_Encoded'] = 0.5  # Default value

        # Apply target encoding for State
        if self.state_encoder is not None and 'State' in encoded:
            try:
                encoded['State_Encoded'] = self.state_encoder.transform(
                    pd.DataFrame({'State': [encoded['State']]})
                ).iloc[0, 0]
            except:
                encoded['State_Encoded'] = 0.5  # Default value

        # Apply label encoding for other categorical features
        for feature in ['Season', 'pH_category', 'crop_category']:
            encoded_col = f'{feature}_Encoded'
            if feature in encoded and encoded_col in self.encoded_features:
                le = self.label_encoders.get(feature)
                if le is not None:
                    try:
                        encoded[encoded_col] = le.transform([encoded[feature]])[0]
                    except:
                        # Handle unseen labels
                        encoded[encoded_col] = 0

        # Create DataFrame with all required features
        df_features = pd.DataFrame([encoded])

        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df_features.columns:
                # Provide reasonable defaults based on feature type
                if 'soil_' in feature:
                    df_features[feature] = 0  # Default for soil features
                elif 'climate_' in feature:
                    if 'rainfall' in feature:
                        df_features[feature] = 500  # Default rainfall
                    elif 'temperature' in feature:
                        df_features[feature] = 25   # Default temperature
                    else:
                        df_features[feature] = 0.5  # Default for other climate features
                elif 'production_' in feature:
                    if 'Area' in feature:
                        df_features[feature] = 1000  # Default area
                    else:
                        df_features[feature] = 2.0   # Default yield stats
                else:
                    df_features[feature] = 0  # Default for encoded features

        return df_features[self.feature_names]

    def predict(self, input_data: Dict, return_confidence: bool = True,
                return_features: bool = False) -> Dict:
        """
        Predict crop yield with enhanced features.

        Args:
            input_data: Dictionary containing feature values
            return_confidence: Whether to return confidence estimation
            return_features: Whether to return the processed features

        Returns:
            Dictionary with prediction and metadata
        """
        try:
            # Calculate derived features
            enhanced_features = self._calculate_derived_features(input_data)

            # Encode features
            X = self._encode_features(enhanced_features)

            # Make prediction
            prediction = self.model.predict(X)[0]

            # Create comprehensive result
            result = {
                'prediction': {
                    'yield_tons_per_ha': float(prediction),
                    'yield_kg_per_ha': float(prediction * 1000),
                    'estimated_production_tons': float(prediction * enhanced_features.get('Area', 1))
                },
                'inputs': input_data,
                'status': 'success',
                'model_info': {
                    'name': self.preprocessing['dataset_info']['best_model'],
                    'r2_score': self.preprocessing['dataset_info']['best_r2'],
                    'mae': self.preprocessing['dataset_info']['best_mae']
                }
            }

            # Add confidence estimation
            if return_confidence:
                confidence = self._estimate_confidence(X, enhanced_features)
                result['confidence'] = confidence

            # Add processed features if requested
            if return_features:
                result['processed_features'] = enhanced_features
                result['feature_vector'] = X.to_dict('records')[0]

            return result

        except Exception as e:
            import traceback
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'inputs': input_data
            }

    def _estimate_confidence(self, X: pd.DataFrame, features: Dict) -> Dict:
        """
        Estimate prediction confidence based on input characteristics.
        """
        confidence_score = 0.8  # Base confidence

        # Adjust based on feature completeness
        missing_features = [f for f in self.feature_names if f not in X.columns]
        if missing_features:
            confidence_score -= 0.1

        # Adjust based on crop-state combination familiarity
        crop = features.get('Crop', '').lower()
        state = features.get('State', '').lower()

        # Common crop-state combinations (simplified)
        common_combinations = [
            ('punjab', 'wheat'), ('haryana', 'rice'), ('maharashtra', 'cotton'),
            ('karnataka', 'coffee'), ('tamil nadu', 'rice'), ('gujarat', 'groundnut')
        ]

        is_common = any(state in combo[0] and crop in combo[1] for combo in common_combinations)
        if not is_common:
            confidence_score -= 0.05

        # Adjust based on extreme values
        if 'pH' in features:
            ph = features['pH']
            if ph < 4.0 or ph > 9.0:
                confidence_score -= 0.05

        if 'temperature' in features:
            temp = features['temperature']
            if temp < 5 or temp > 45:
                confidence_score -= 0.05

        # Ensure confidence bounds
        confidence_score = max(0.5, min(0.95, confidence_score))

        # Determine level
        if confidence_score >= 0.85:
            level = "HIGH"
        elif confidence_score >= 0.75:
            level = "MEDIUM-HIGH"
        elif confidence_score >= 0.65:
            level = "MEDIUM"
        elif confidence_score >= 0.55:
            level = "MEDIUM-LOW"
        else:
            level = "LOW"

        return {
            'score': confidence_score,
            'level': level,
            'margin_of_error': f"±{((1-confidence_score)*100):.0f}%",
            'factors_considered': ['feature_completeness', 'crop_state_familiarity', 'value_ranges']
        }

    def predict_batch(self, input_list: List[Dict]) -> List[Dict]:
        """
        Predict for multiple inputs.
        """
        return [self.predict(data, return_confidence=False) for data in input_list]

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance from the model.
        """
        if hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
            importances = self.model.named_steps['regressor'].feature_importances_
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
        else:
            return pd.DataFrame(columns=['feature', 'importance'])

    def explain_prediction(self, input_data: Dict) -> Dict:
        """
        Provide explanation for a prediction.
        """
        result = self.predict(input_data, return_features=True)

        if result['status'] != 'success':
            return result

        explanation = {
            'prediction': result['prediction'],
            'key_factors': [],
            'recommendations': []
        }

        features = result.get('processed_features', {})

        # Analyze soil factors
        if 'pH' in features:
            ph = features['pH']
            if ph < 5.5:
                explanation['key_factors'].append(f"Low pH ({ph}) may limit nutrient availability")
                explanation['recommendations'].append("Consider lime application to raise pH")
            elif ph > 7.5:
                explanation['key_factors'].append(f"High pH ({ph}) may cause nutrient deficiencies")
                explanation['recommendations'].append("Consider sulfur application to lower pH")

        # Analyze nutrient balance
        if all(x in features for x in ['N', 'P', 'K']):
            n, p, k = features['N'], features['P'], features['K']
            ideal_ratio = (4, 2, 1)  # Simplified ideal N:P:K ratio

            actual_ratio = (n/ideal_ratio[0], p/ideal_ratio[1], k/ideal_ratio[2])
            imbalance = max(actual_ratio) / min(actual_ratio) if min(actual_ratio) > 0 else 10

            if imbalance > 3:
                explanation['key_factors'].append(f"Nutrient imbalance detected (N:{n}, P:{p}, K:{k})")
                explanation['recommendations'].append("Consider balanced fertilizer application")

        # Analyze climate factors
        if 'temperature' in features and 'Crop' in features:
            temp = features['temperature']
            crop = features['Crop']

            optimal_ranges = {
                'rice': (20, 35), 'wheat': (10, 25), 'maize': (18, 32)
            }

            for crop_key, (min_temp, max_temp) in optimal_ranges.items():
                if crop_key in crop.lower():
                    if temp < min_temp:
                        explanation['key_factors'].append(f"Temperature ({temp}°C) below optimal for {crop}")
                        explanation['recommendations'].append(f"Consider later planting or cold-tolerant varieties")
                    elif temp > max_temp:
                        explanation['key_factors'].append(f"Temperature ({temp}°C) above optimal for {crop}")
                        explanation['recommendations'].append(f"Consider heat-tolerant varieties or shading")
                    break

        return explanation


# Example usage
if __name__ == "__main__":
    # Example initialization
    predictor = EnhancedIndiaCropPredictor(
        model_path=model_filename,
        preprocessing_path=preprocessing_filename
    )

    print("\n🧪 TESTING ENHANCED PREDICTOR:")
    print("="*50)

    # Test cases with comprehensive features
    test_cases = [
        {
            "State": "Punjab",
            "Crop": "Wheat",
            "Season": "Rabi",
            "N": 120,
            "P": 40,
            "K": 30,
            "pH": 6.8,
            "rainfall": 450,
            "temperature": 22,
            "Area": 1000
        },
        {
            "State": "Maharashtra",
            "Crop": "Cotton",
            "Season": "Kharif",
            "N": 80,
            "P": 30,
            "K": 25,
            "pH": 7.2,
            "rainfall": 600,
            "temperature": 28,
            "Area": 500
        },
        {
            "State": "Karnataka",
            "Crop": "Rice",
            "Season": "Kharif",
            "N": 100,
            "P": 35,
            "K": 20,
            "pH": 5.8,
            "rainfall": 800,
            "temperature": 26,
            "Area": 750
        }
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n📊 Test {i}: {test_input['Crop']} in {test_input['State']}")

        # Get prediction with explanation
        result = predictor.predict(test_input, return_confidence=True)

        if result["status"] == "success":
            print(f"   Yield Prediction: {result['prediction']['yield_tons_per_ha']:.3f} tons/ha")
            print(f"   Confidence: {result['confidence']['level']}")

            # Get explanation
            explanation = predictor.explain_prediction(test_input)
            if explanation['key_factors']:
                print(f"   Key Factors: {', '.join(explanation['key_factors'][:2])}")
        else:
            print(f"   Error: {result['error']}")
