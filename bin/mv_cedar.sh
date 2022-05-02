#!/bin/bash
. /home/fwfop/.bashrc

scp -r /bluesky/fireweather/exburn/burns/inputs/input_sounding.$(date '+%Y%m%d12') rodell@cedar.computecanada.ca:/home/rodell/projects/def-rstull/shared_data/