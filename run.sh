#!/bin/bash
docker run --runtime=nvidia -it -u $(id -u):$(id -g) -v $PWD:/home/app/ marian bash -c "cd /home/app; bash"
