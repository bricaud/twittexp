# twittexp
Twitter exploration code. collecting and analyzing tweets, retweets, mentions.

This project  implement the concept of *spikyball sampling* are variant of the *snowball sampling* on the social network Twitter. Starting from one or a few user account, the program will collect their latest tweets, extract the retweets/mentions and follow them. It expands like a snowball from user mentioned to user mentioned. The spiky version take a random subset of the mentions at each step to limit the number of users to explore.

After the collection of users and mentions, a graph of users is made using the python module `networkx`. Info from the tweets like text and hashtags are collected.

This project uses the `pysad` module available at [bricaud/pysad](https://github.com/bricaud/pysad). This is where the *spikyball sampling* is implemented. Clone it and add the path where to find it in the `config.json` configuration file.

The main file is the Jupyter notebook `collect_analyze_tweets.ipynb`.

Don't forget to add your Twitter developper credentials in the `twitter_credentials.json` . An empty model is given with `twitter_credentials_empty.json`. You have to apply for a [developer account](https://developer.twitter.com/en/apply-for-access).

The file `initial_accounts.json` contains the list of initial users from where the exploration starts, for different topics.
The notebook `add_initial_accounts.ipynb` give example on how to add new list of users to explore for the `initial_accounts.txt`.