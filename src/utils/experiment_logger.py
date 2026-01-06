import pickle
import os
import uuid
import datetime

class ResultLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def save(self, data, name_prefix="result"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{name_prefix}_{timestamp}_{unique_id}.pkl"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
            
        print(f"Saved results to {filepath}")
        return filepath
