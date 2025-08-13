GENERATE=TRUE  # set to TRUE only if you want to regenerate data.csv files
if [ "$GENERATE" = "TRUE" ]; then
    mkdir -p reproduce/
    python3 ./case.py 101 20 50 4.0 flat ratios.txt reproduce/bflat2000m/
    python3 ./case.py 101 20 50 4.0 rough ratios.txt reproduce/brough2000m/
    python3 ./case.py 201 30 200 1.0 flat ratios.txt reproduce/bflat1000m/
    python3 ./case.py 201 30 200 1.0 rough ratios.txt reproduce/brough1000m/
    python3 ./case.py 401 40 400 0.5 flat ratios.txt reproduce/bflat500m/
    python3 ./case.py 401 40 400 0.5 rough ratios.txt reproduce/brough500m/
fi
python3 ./histogram.py reproduce/bflat2000m/NOREG/data.csv bflat2000mNOREG.png
python3 ./histogram.py reproduce/bflat2000m/REG__/data.csv bflat2000mREG.png
python3 ./histogram.py reproduce/brough2000m/NOREG/data.csv brough2000mNOREG.png
python3 ./histogram.py reproduce/brough2000m/REG__/data.csv brough2000mREG.png
python3 ./histogram.py reproduce/bflat1000m/NOREG/data.csv bflat1000mNOREG.png
python3 ./histogram.py reproduce/bflat1000m/REG__/data.csv bflat1000mREG.png
python3 ./histogram.py reproduce/brough1000m/NOREG/data.csv brough1000mNOREG.png
python3 ./histogram.py reproduce/brough1000m/REG__/data.csv brough1000mREG.png
python3 ./histogram.py reproduce/bflat500m/NOREG/data.csv bflat500mNOREG.png
python3 ./histogram.py reproduce/bflat500m/REG__/data.csv bflat500mREG.png
python3 ./histogram.py reproduce/brough500m/NOREG/data.csv brough500mNOREG.png
python3 ./histogram.py reproduce/brough500m/REG__/data.csv brough500mREG.png

