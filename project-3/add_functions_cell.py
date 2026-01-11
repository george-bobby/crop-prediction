import nbformat

# Read the notebook
nb = nbformat.read('main.ipynb', as_version=4)

# The code for synthetic functions
func_code = """# Helper functions to create synthetic data when datasets are not available
from datetime import datetime, timedelta

def create_synthetic_crop_production_data(num_records=1000):
    '''Generate synthetic crop production data'''
    np.random.seed(42)
    
    states = ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Gujarat', 'Punjab', 'Uttar Pradesh', 'Madhya Pradesh']
    districts = ['District_' + str(i) for i in range(1, 21)]
    crops = ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize', 'Jowar', 'Bajra', 'Pulses', 'Groundnut', 'Soyabean']
    seasons = ['Kharif', 'Rabi', 'Summer', 'Whole Year']
    years = list(range(2010, 2024))
    
    data = {
        'State_Name': np.random.choice(states, num_records),
        'District_Name': np.random.choice(districts, num_records),
        'Crop_Year': np.random.choice(years, num_records),
        'Season': np.random.choice(seasons, num_records),
        'Crop': np.random.choice(crops, num_records),
        'Area': np.random.uniform(100, 10000, num_records),
        'Production': np.random.uniform(500, 50000, num_records)
    }
    
    return pd.DataFrame(data)

def create_synthetic_crop_recommendation_data(num_records=2200):
    '''Generate synthetic crop recommendation data'''
    np.random.seed(42)
    
    crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 
             'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 
             'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
    
    data = {
        'N': np.random.randint(0, 140, num_records),
        'P': np.random.randint(5, 145, num_records),
        'K': np.random.randint(5, 205, num_records),
        'temperature': np.random.uniform(8, 45, num_records),
        'humidity': np.random.uniform(14, 100, num_records),
        'ph': np.random.uniform(3.5, 9.5, num_records),
        'rainfall': np.random.uniform(20, 300, num_records),
        'label': np.random.choice(crops, num_records)
    }
    
    return pd.DataFrame(data)

def create_synthetic_market_price_data(num_records=500):
    '''Generate synthetic market price data'''
    np.random.seed(42)
    
    states = ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Gujarat', 'Punjab', 'Haryana', 'Madhya Pradesh']
    districts = ['District_' + str(i) for i in range(1, 11)]
    markets = ['Market_' + str(i) for i in range(1, 6)]
    commodities = ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize', 'Onion', 'Potato', 'Tomato']
    varieties = ['FAQ', 'Grade A', 'Grade B', 'Medium', 'Large', 'Other']
    grades = ['FAQ', 'Grade A', 'Grade B']
    
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_records)]
    
    data = {
        'State': np.random.choice(states, num_records),
        'District': np.random.choice(districts, num_records),
        'Market': np.random.choice(markets, num_records),
        'Commodity': np.random.choice(commodities, num_records),
        'Variety': np.random.choice(varieties, num_records),
        'Grade': np.random.choice(grades, num_records),
        'Arrival_Date': np.random.choice(dates, num_records),
        'Min Price': np.random.uniform(1000, 4000, num_records),
        'Max Price': np.random.uniform(4000, 8000, num_records),
        'Modal Price': np.random.uniform(2000, 6000, num_records)
    }
    
    df = pd.DataFrame(data)
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
    df['year'] = df['Arrival_Date'].dt.year
    df['month'] = df['Arrival_Date'].dt.month
    df['day'] = df['Arrival_Date'].dt.day
    
    return df

print("✓ Synthetic data generation functions loaded")"""

# Insert after cell 6 (imports cell)
cell_index = 6

if True:
    # Create new code cell
    new_cell = nbformat.v4.new_code_cell(source=func_code)
    # Insert it after the imports cell
    nb.cells.insert(cell_index + 1, new_cell)
    # Save the notebook
    nbformat.write(nb, 'main.ipynb')
    print(f"✓ Successfully added synthetic functions cell after cell index {cell_index}")
else:
    print("✗ Could not find the imports cell")
