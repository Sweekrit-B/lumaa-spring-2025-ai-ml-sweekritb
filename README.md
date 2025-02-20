# Scientific Paper Content Recommendation System

**Author**: Sweekrit Bhatnagar

Upon search input from a user, returns the top search results based on scientific article abstracts.

## Dataset

The public domain dataset comes from Kaggle: https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts. Originally meant for paper classification systems, this dataset, containing paper titles, abstracts, and subject categories has been repurposed for this reccomendation system.

NOTE: within `paper_recc.py`, the `filter_kaggle_data(path, new_file)` function takes the Kaggle data and filters it to retrieve the first 500 rows, as specified by the requirements of this project. However, users could edit these values and grab variable amounts (or all the data) for their purposes.

## Setup

### Fetch from Github

Click on the green **<> Code** button on the top right corner, where you will see an option to clone the dataset. Select HTTPS, and click the copy button on the right side of the text box.

Navigate to your project directory in your terminal, and clone the dataset.

```
cd C:\path\to\your\project
git clone https://github.com/Sweekrit-B/lumaa-spring-2025-ai-ml-sweekritb.git
```

### Virtual Environment Setup

Once inside your project directory, run the following commands to setup your virual environment and activate it:

For Windows:

```bash Windows
py -m venv venv
.\venv\Scripts\activate
```

For MacOS/Linux:

```bash MacOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Your terminal prompt should change to show the virtual environment name.

From here, make sure to install all necessary dependencies.

```
pip install -r requirements.txt
```

This will install all necessary packages listed in the `requirements.txt` of this file.

### Running the Code

To run the code without a frontend UI, use the following format:

```
py .\paper_recc.py "<input>" <number of top results>
```

This will return a dataframe in your terminal containing the titles of all the names of the relevant articles.

However, there is also an option to run it with a frontend UI. In this case, please run the following:

```
py -m streamlit run \streamlit_ui.py
```

This will take you to a new page in your localhost, where you will find a UI where you can search for papers and view the results.

### Results

In either running method, the code uses TF-IDF vectorization and cosine similarity to parse through paper abstracts and determine what is most relevant to user input. Both return a dataframe with the top number of search results (as specified by the user) with the title and abstract - however, the frontend UI has it formatted more nicely and is more relevant for usage.

### Salary Expectations

$2000/month, or $25/hr working 20 hours/week.
