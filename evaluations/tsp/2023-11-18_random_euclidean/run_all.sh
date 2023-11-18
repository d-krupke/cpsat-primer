# Generate the instances 
python3 00_generate_instances.py && \
# Run the benchmark
python3 01_run_benchmark.py && \
# Extract the data into a nice table
python3 03_extract_data.py && \
# Plot the data by running the notebook
jupyter nbconvert --to notebook --execute 04_analyze.ipynb
