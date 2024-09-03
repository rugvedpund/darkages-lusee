#!/usr/bin/bash

for aa in {10..80..10}
do
	python scripts/runLuSEEsim.py data/configs/a${aa}_dt3600.yaml
done
