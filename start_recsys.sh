#!/usr/bin/env bash
~/spark-2.1.0-bin-hadoop2.7/bin/spark-submit --master local[3] --total-executor-cores 5 --executor-memory 8g music_recsys.py
