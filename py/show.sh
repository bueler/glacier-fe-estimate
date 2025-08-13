GENERATE=FALSE  # set to TRUE only if you want to regenerate data.csv files
if [ "$GENERATE" = "TRUE" ]; then
    python3 ./case.py 201 15 50 1.0 smooth ratios.txt result/
fi
python3 ./histogram.py result/NOREG/data.csv smooth2000mNOREG.png
python3 ./histogram.py result/REG__/data.csv smooth2000mREG.png
