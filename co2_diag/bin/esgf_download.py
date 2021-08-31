#!/usr/bin/env python
###############################################
# Download CMIP6 data containing co2
# Author: D.E.Kaufman (daniel.kaufman@pnnl.gov)
# Created: 2021-06-18
###############################################
import os, tempfile, subprocess

from pyesgf.search import SearchConnection

# Initialize the search with a specific data node
conn = SearchConnection('https://esgf-data.dkrz.de/esg-search', distrib=True)

# Specify our search criteria
ctx = conn.new_context(project='CMIP6', experiment_id='esm-hist', variable='co2, zg', frequency='mon', realm='atmos')

# Find out how many hits our search will retrieve
print(f"Hit count = {ctx.hit_count}")

# Execute the search
results = ctx.search()

#hits = []
#for r in results:
#    hits.append(r.file_context().search())

# Download a file using the ESGF wget script extracted from the server:
ds = ctx.search()[0]
fc = ds.file_context()
wget_script_content = fc.get_download_script()
file_descriptor, script_path = tempfile.mkstemp(suffix='.sh', prefix='download-')
with open(script_path, "w") as writer:
    writer.write(wget_script_content)
os.close(file_descriptor)
print(script_path)

os.chmod(script_path, 0o750)
download_dir = os.path.dirname(script_path)
subprocess.check_output("{}".format(script_path), cwd=download_dir)

print(download_dir)
print('Done.')
