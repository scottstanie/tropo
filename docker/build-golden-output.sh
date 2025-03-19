#!/bin/bash

# Step 1: Create directories
mkdir -p delivery_data_tropo/configs
mkdir -p delivery_data_tropo/input_data
mkdir -p delivery_data_tropo/golden_output

# Step 2: Navigate to delivery_data_tropo and download data
cd delivery_data_tropo || exit 1
opera_tropo download -o input_data -s3 opera-dev-lts-fwd-hyunlee --date 20190613 --hour 06

# Step 3: Create configuration file
CONFIG_FILE="configs/runconfig_20190613_06.yaml"
opera_tropo config -input input_data/D06130600061306001.zz.nc -out golden_output/ -c "$CONFIG_FILE"

# Step 4: Run the opera_tropo process
opera_tropo run "$CONFIG_FILE"

# Step 5: Rename the output files
for file in golden_output/*.nc; do
    timestamp=$(basename "$file" | sed -E 's/.*([0-9]{8}T[0-9]{2}).*/\1/')
    mv "$file" "golden_output/golden_output_$(basename "$file" .nc).nc"
done

for file in golden_output/*.png; do
    timestamp=$(basename "$file" | sed -E 's/.*([0-9]{8}T[0-9]{2}).*/\1/')
    mv "$file" "golden_output/golden_output_$(basename "$file" .png).png"
done
