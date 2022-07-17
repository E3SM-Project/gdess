#!/bin/bash

echo
echo 'Building API reference docs'
echo
mkdir -p "./docs"
pushd "./docs/.." >/dev/null
pdoc3 --html \
      --force \
      --config latex_math=True \
      --output-dir "./docs" \
      gdess
popd >/dev/null

## D.E.K. - add this line to the top of the index.html...
## <meta http-equiv="refresh" content="0; url=https://e3sm-project.github.io/gdess/" />

echo
echo "All good. Docs in: ./docs"
echo
echo "    file://docs/index.html"
echo