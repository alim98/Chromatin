Full Sample Tracking:
Each sample from the CSV file (like "1,Glia,0011") is tracked as a "full sample"
We store metadata for these full samples in self.sample_metadata
Subsample/Crop Management:
Each full sample is virtually divided into multiple 80×80×80 sub-volumes
We create a mapping from dataset index to (sample_idx, subsample_idx) in self.subsample_mapping
This allows efficient access to any specific crop from any sample
Detailed Crop Information:
Each returned sample now includes metadata about which crop it is and where it came from
This helps track the origin of predictions in your model
Memory Efficiency:
Volumes are only loaded when accessed, not up front
After processing, volumes are explicitly deleted to free memory
Better Diagnostics:
We show how many full samples are found
We estimate and report the total number of sub-volumes across all samples
How It Works In Practice
When the dataset is initialized, it estimates how many crops each sample will yield
It creates a mapping from dataset indices to specific crops in specific samples
When __getitem__(idx) is called, it:
Maps the index to the correct sample and crop
Loads the full volume if needed
Extracts just the requested crop
Returns that specific crop with detailed metadata