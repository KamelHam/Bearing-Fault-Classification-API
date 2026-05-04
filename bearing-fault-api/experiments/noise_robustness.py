import numpy as np
import pandas as pd
from typing import Dict, List
import asyncio
import aiohttp
import json

class NoiseRobustnessTester:
    """Test model robustness under different noise levels."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        
    def add_noise(self, df: pd.DataFrame, noise_level: float) -> pd.DataFrame:
        """Add Gaussian noise to data."""
        noisy_df = df.copy()
        for col in df.columns:
            noise = np.random.normal(0, noise_level * df[col].std(), len(df))
            noisy_df[col] = df[col] + noise
        return noisy_df
    
    async def test_noise_level(self, session, original_file, noise_level: float) -> Dict:
        """Test a specific noise level."""
        df = pd.read_excel(original_file)
        noisy_df = self.add_noise(df, noise_level)
        
        # Save temporarily
        temp_path = f"/tmp/noisy_{noise_level}.xlsx"
        noisy_df.to_excel(temp_path, index=False)
        
        async with session.post(
            f"{self.api_url}/predict",
            data={'model_name': 'rf'},
            files={'file': open(temp_path, 'rb')}
        ) as response:
            result = await response.json()
            
        return {"noise_level": noise_level, "result": result}
    
    async def run_tests(self, file_path: str, noise_levels: List[float] = [0, 0.05, 0.1, 0.2, 0.3]):
        """Run robustness tests."""
        async with aiohttp.ClientSession() as session:
            tasks = [self.test_noise_level(session, file_path, nl) for nl in noise_levels]
            results = await asyncio.gather(*tasks)
        return results