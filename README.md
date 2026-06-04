# FISH_Filter

## Introduction
This RS-FISH filter allows for spots that were detected outside of the region of interest to be eliminated. The region of interest can be drawn through the application and with the inputted csv file, it will eliminate all points falling outside of the region of interest. 

### Setup

1. Open terminal and navigate to any directory you'd like the application to be. 


```bash
cd ./{DESIRED PATH}
```

2. clone the repository.
```bash
git clone https://github.com/jmo7728/FISH_Filter
cd FISH_Filter
```

3. create a virtual enviornment (Make sure python is installed and pip install pipenv is done)
```bash
pipenv shell
```

4. Install dependencies:
```bash
pipenv install
```

5. Run the application:
```bash
pipenv run python filter_spots.py --image {PATH TO TIF IMG} --csv {PATH TO csv FILE}
```

Example:
```bash
pipenv run python filter_spots.py --image ./tif_images/W1_C=1.tif --csv ./data/trial2.csv 
```