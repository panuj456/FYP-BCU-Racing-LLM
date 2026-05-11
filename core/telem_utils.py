import os
import pickle
import networkx as nx
import re
from typing import List, Dict, Any
import networkx as nx

class TelemUtils:

    @staticmethod
    def load_all_session_graphs(data_path: str) -> dict:
        """
        Scans the data directory for pickle files and loads them into memory.
        """
        graphs = {}
        
        # Check if the directory exists first
        if not os.path.exists(data_path):
            print(f"Error: Data path '{data_path}' does not exist.")
            return graphs
    
        # Iterate through every file in the /data folder
        for filename in os.listdir(data_path):
            # We only care about the .pkl files you generated earlier
            if filename.endswith(".pkl"):
                file_path = os.path.join(data_path, filename)
                
                # Extract the session ID from the filename
                # Example: 'session4_stg.pkl' -> 'session4'
                session_id = filename.replace("_stg.pkl", "").replace(".pkl", "")
                
                try:
                    with open(file_path, 'rb') as f:
                        # Load the NetworkX DiGraph object
                        G = pickle.load(f)
                        
                        # Store it in our dictionary
                        if isinstance(G, nx.Graph):
                            graphs[session_id] = G
                            print(f"Successfully loaded {session_id} ({len(G.nodes)} nodes)")
                        else:
                            print(f"Warning: {filename} did not contain a valid Graph object.")
                
                except Exception as e:
                    print(f"Failed to load {filename}: {str(e)}")
                    
        return graphs
