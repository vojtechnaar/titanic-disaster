# Titanic Disaster: Survival Prediction in Python and R

This repository predicts which passengers survived the sinking of the Titanic (April 15, 1912) using
**logistic regression**, implemented twice, each in its own Docker container:

| Container | Language | Model | Folder |
|-----------|----------|-------|--------|
| `titanic-python` | Python 3.12 | scikit-learn `LogisticRegression` | [`src/python/`](src/python) |
| `titanic-r` | R 4.4.2 | `glm(family = binomial)` | [`src/r/`](src/r) |

Each script loads the Kaggle data, explores it, cleans it and engineers features, trains the model,
reports training accuracy, predicts on the test set, reports test accuracy, and saves the predictions.
**Every step prints what it is doing**, so the whole analysis can be followed in the terminal without
opening the code.

The dataset is **not** included in this repository. Step 2 below shows how to download it.

---

## Quick start (TL;DR)

Assuming Git and Docker are installed and the three Kaggle CSV files are already in `src/data/`:

```bash
git clone https://github.com/vojtechnaar/titanic-disaster.git
cd titanic-disaster
# put train.csv, test.csv and gender_submission.csv into src/data/  (see Step 2)

docker build -t titanic-python src/python
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python

docker build -t titanic-r src/r
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

The detailed version of each step follows.

---

## Step 0: Prerequisites

You need two tools installed:

1. **Git**: <https://git-scm.com/downloads> (check with `git --version`)
2. **Docker Desktop**: <https://www.docker.com/products/docker-desktop/> (check with `docker --version`)

**Docker Desktop must be running** before you use any `docker` command. On macOS and Windows, open the
Docker Desktop app and wait until it reports that the engine is running.

You also need a free **Kaggle account** to download the data: <https://www.kaggle.com/account/login>.

Python and R do **not** need to be installed on your machine. Everything runs inside the containers.

## Step 1: Clone the repository

```bash
git clone https://github.com/vojtechnaar/titanic-disaster.git
cd titanic-disaster
```

**Run every remaining command from this `titanic-disaster` folder (the repository root).**

## Step 2: Download the data

The scripts expect these three files from the Kaggle Titanic competition inside `src/data/`:

```
src/data/train.csv               # 891 passengers, includes the Survived column
src/data/test.csv                # 418 passengers, no Survived column
src/data/gender_submission.csv   # Kaggle's sample submission (used as test-set reference labels)
```

The `src/data/` folder already exists in the repository (it is empty and git-ignored).

### Option A: download in the browser (easiest)

1. Sign in to Kaggle and open <https://www.kaggle.com/competitions/titanic/data>.
2. If Kaggle asks you to, click **Join Competition** / **I Understand and Accept**. Kaggle only allows
   downloads after you accept the competition rules.
3. Scroll down and click **Download All**. This downloads `titanic.zip`.
4. Unzip it and move the three CSV files into `src/data/`. For example, if the zip is in your
   Downloads folder:

   ```bash
   # macOS / Linux
   unzip ~/Downloads/titanic.zip -d src/data
   ```

   ```powershell
   # Windows PowerShell
   Expand-Archive -Path "$HOME\Downloads\titanic.zip" -DestinationPath src\data
   ```

### Option B: download with the Kaggle CLI

```bash
pip install kaggle
# Create an API token at https://www.kaggle.com/settings -> "Create New Token" and save the
# downloaded kaggle.json to ~/.kaggle/kaggle.json  (Windows: C:\Users\<you>\.kaggle\kaggle.json)
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip -d src/data
```

### Check the files are in place

```bash
ls src/data
# expected: gender_submission.csv  test.csv  train.csv  (plus titanic.zip if you used Option B)
```

## Step 3: Run the Python container

**Build the image.** This installs the packages pinned in `src/python/requirements.txt`. It takes
about 30 to 60 seconds the first time.

```bash
docker build -t titanic-python src/python
```

**Run the container.** The `-v` flag mounts your local `src/data` folder into the container at
`/app/data`, so the script can read the CSVs and write its predictions back to your machine.

```bash
# macOS / Linux / Git Bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```

```powershell
# Windows PowerShell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-python
```

```bat
:: Windows Command Prompt (cmd.exe)
docker run --rm -v "%cd%\src\data:/app/data" titanic-python
```

## Step 4: Run the R container

**Build the image.** This runs `src/r/install_packages.R`, which installs `dplyr` and `readr`. On an
Intel/AMD machine this takes about 1 minute. On Apple Silicon (M1/M2/M3/M4) the packages are compiled
from source, so the first build takes about 3 minutes. Later builds are cached and take seconds.

```bash
docker build -t titanic-r src/r
```

**Run the container**, using the same `-v` mount as for Python:

```bash
# macOS / Linux / Git Bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

```powershell
# Windows PowerShell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-r
```

```bat
:: Windows Command Prompt (cmd.exe)
docker run --rm -v "%cd%\src\data:/app/data" titanic-r
```

## Step 5: Read the output

Both scripts print the same five steps to the terminal:

| Step | What is printed |
|------|-----------------|
| 1. Explore `train.csv` | first rows, column types, missing values, survival rate overall, by sex and by class |
| 2. Clean & engineer features | one line per change, each with a count of rows affected (see the table below) |
| 3. Train logistic regression | features used, model coefficients, **training accuracy**, 5-fold cross-validated accuracy |
| 4. Predict on `test.csv` | the same cleaning steps applied to the test set, number of predicted survivors |
| 5. Test-set accuracy | **test accuracy**, followed by a final summary |

Both runs also save their predictions in Kaggle's submission format (`PassengerId,Survived`):

- `src/data/predictions_python.csv`
- `src/data/predictions_r.csv`

### Results

| | Python | R |
|---|---|---|
| Training accuracy | 0.8395 | 0.8384 |
| 5-fold cross-validated accuracy | 0.8350 | 0.8339 |
| Test accuracy (vs `gender_submission.csv`) | 0.9187 | 0.9187 |

> **About test accuracy:** Kaggle hides the true `Survived` labels for `test.csv`. The scripts measure
> test accuracy against `gender_submission.csv`, Kaggle's sample answer in which every woman survives
> and every man dies. That number shows how closely the model agrees with this simple baseline. It is
> not the true accuracy. The 5-fold cross-validated accuracy on the training data is the honest
> estimate of performance on unseen passengers. To get the official score, upload
> `predictions_python.csv` or `predictions_r.csv` to the
> [competition's Submit page](https://www.kaggle.com/competitions/titanic/submissions).

---

## What the scripts do to the data

The same changes are applied in both languages. All fill values (medians, most common port) are learned
from `train.csv` only and then reused on `test.csv`.

| Change | Why |
|--------|-----|
| `Title` extracted from `Name` (Mr, Mrs, Miss, Master, Rare) | A strong signal: "Master" marks young boys, "Mrs" married women |
| Missing `Age` (177 train / 86 test) filled with the median age of the passenger's title | Better than one global median, because a "Master" is a child |
| Missing `Embarked` (2 train) filled with the most common port, `S` | Only 2 rows are affected |
| Missing `Fare` (1 test) filled with the training median | Only 1 row is affected |
| `HasCabin` = 1 if a cabin is recorded, then `Cabin` dropped | `Cabin` is 77% missing, but whether a cabin is recorded still predicts survival |
| `FamilySize` = `SibSp` + `Parch` + 1, and `IsAlone` | Combines two related columns into one |
| `Sex` encoded as 0 = female, 1 = male | Needed for the model |
| Titles `Mr` and `Mrs` merged into `Mr/Mrs` before modelling | Once `Sex` is a feature, they add nothing: every Mr is male and every Mrs is female. Keeping both makes `Sex` and `Title` redundant, and R's unregularized `glm` then produces exploding coefficients. |
| `Pclass`, `Embarked`, `Title` one-hot encoded (Python) / factors (R) | Treats them as categories rather than numbers |
| `PassengerId`, `Name`, `Ticket`, `Cabin`, `SibSp`, `Parch` dropped | IDs, free text, or replaced by the features above |

Model: Python standardizes the features and fits an L2-regularized `LogisticRegression`. R fits an
unregularized `glm(..., family = binomial)`. Both classify a passenger as a survivor when the predicted
probability is above 0.5.

---

## Repository structure

```
titanic-disaster/
├── .github/
│   └── CODEOWNERS            # makes the repository owner the code owner of every file
├── .gitignore                # ignores src/data/, .DS_Store, virtual envs, R session files
├── README.md
└── src/
    ├── data/                 # put the Kaggle CSVs here (git-ignored, never committed)
    ├── python/
    │   ├── Dockerfile        # python:3.12.8-slim, installs requirements, runs the script
    │   ├── requirements.txt  # pinned package versions
    │   └── titanic_model.py
    └── r/
        ├── Dockerfile        # rocker/r-ver:4.4.2, runs install_packages.R, then the script
        ├── install_packages.R
        └── titanic_model.R
```

### Package versions

- **Python** (`requirements.txt`): numpy 2.2.1, pandas 2.2.3, scikit-learn 1.6.0, with their
  dependencies pinned too.
- **R** (`install_packages.R`): dplyr 1.1.4, readr 2.1.5. The `rocker/r-ver:4.4.2` image points CRAN
  at a dated snapshot (2025-02-27), so the versions are fixed, and the install script fails the build
  if a different version gets installed.

### Branches

- `main`: the final, reviewed version (what you get when you clone)
- `develop`: integration branch; every feature is merged here first through a pull request
- `feature/titanic-model`: branched off `develop`; where the code was written

Changes went `feature/titanic-model` → `develop` → `main` through pull requests. They are listed in the
[Pull requests tab](https://github.com/vojtechnaar/titanic-disaster/pulls?q=is%3Apr).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot connect to the Docker daemon` / `failed to connect to the docker API` | Docker Desktop is not running. Start it, wait for the engine, then retry. |
| `ERROR: could not find /app/data/train.csv` | The CSVs are not in `src/data/`, or the command was not run from the repo root. Check with `ls src/data`, and make sure you are in the `titanic-disaster` folder. |
| Windows: `invalid reference format` or the mount is empty | Use the PowerShell or cmd variant of the `docker run` command above, not the `$(pwd)` one. |
| Kaggle returns `403 Forbidden` | You have not accepted the competition rules. Open the competition page and click **Join Competition**. |
| R build seems slow on a Mac | Expected on Apple Silicon (packages compile from source). Only the first build is slow. |
