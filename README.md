# NFL Fantasy Points Prediction Based on Prior Season Stats

## Table of Contents
- [Overview](#project-overview)
- [Data/Sources](#data-sources)
    - [Identifier columns](#identifier-columns)
    - [Target Variable](#target-variable)
    - [Lagged Stats columns](#lagged-stats-columns)
    - [Lagged Stats Basis columns](#lagged-stats-basis-columns)
- [Methodology](#methodology)
    - [Data Preparation/Cleaning](#data-preparationcleaning)
    - [Feature Engineering](#feature-engineering)
    - [Limitations/Assumptions](#limitationsassumptions)
    - [Modeling](#modeling)
- [Results](#results)
    - [SHAP Tests](#shap-tests)
    - [Top 10s](#top-10s)
    - [Gemini Analysis](#gemini-analysis)
- [Next Steps](#next-steps)

## Project Overview
There are innumerable strategies to drafting a league-winning fantasy football team. Some love crafting the perfect big board after cranking out countless mock drafts, others love simply drafting their favorite players, and others let fate (autodraft) take the wheel. In today's world of AI, fantasy football enthusiasts who also happen to be data professionals, such as myself, are definitely curious of ways data can unlock the secret to winning bragging rights and the champion's prize (and more importantly, avoid the last-place punishment). 

This project seeks to predict 2025 weekly fantasy football points (PPR scoring) for quarterbacks, wide receivers, tight ends, and running backs (including fullbacks) using the prior season's stats as inputs for training and prediction into machine learning and deep learning models. 

*Disclaimer:* this project seeks only to see what 2025 weekly fantasy football points would be given just the prior season's stats as features out of curiosity of how well these features could perform. It is obvious that this project lacks many useful features, as reflected in the performance, testing on the features used, and domain knowledge. More on this in the [Methodology](#methodology) section.

## Data/Sources
The following are sources of data used in this project:
- **nfl_data_py** (package): individual player weekly stats, weekly team matchup data (through 2024), and seasonal roster data
- **Fixture Download** (downloaded CSV from https://fixturedownload.com/results/nfl-2025?utm_source=chatgpt.com): 2025 weekly team matchup data

Using the above sources, historical dataframes for training and testing as well as a scoring dataframe for 2025 predictions were assembled. Each row corresponds to a player's PPR fantasy points and lagged stats in a given week and season. More on the lagged stats in the [Methodology](#methodology) section. The dataframes contained the following columns:
### Identifier columns
- **player_id**: player's unique ID
- **player_name**: player's full name
- **game_number**: the game number of the season of the player's team (e.g. if a player's team has a bye week on week 9, then week 10 would be game number 9)
- **position**: the player's (abbreviated) primary position
- **team**: the player's (abbreviated) team
- **season**: the season that the row corresponds to
- **week**: the week that the row corresponds to
- **opponent_team**: the opponent team of the corresponding week that the player was facing
- **day_slate**: the day of the week and time slate (morning, afternoon, night, or "global" for global games due to unordinary times) of the player's game in the given week and season
- **location**: the (abbreviated) home team in the given week and season ("global" for global games)
### Target Variable
- **fantasy_points_ppr**: the PPR points the player scored in the given week and season
### Lagged Stats columns
- **completions_lagged**: the number of complete passes by a player, lagged
- **attempts_lagged**: the number of pass attempts by a player, lagged
- **passing_yards_lagged**: the number of passing yards by a player, lagged
- **passing_tds_lagged**: the number of passing touchdowns by a player, lagged
- **interceptions_lagged**: the number of interceptions thrown by a player, lagged
- **sacks_lagged**: the number of sacks a player takes, lagged
- **sack_yards_lagged**: the number of sack yards a player takes, lagged
- **sack_fumbles_lagged**: the number of fumbles caused by sacks the player has, lagged
- **sack_fumbles_lost_lagged**: the number of fumbles caused by sacks that a player *loses*, lagged
- **passing_air_yards_lagged**: the number of passing yards in the air that a player has, lagged
- **passing_yards_after_catch_lagged**: the number of passing yards that a player has after a pass is caught, lagged
- **passing_first_downs_lagged**: the number of first downs a player gets via a pass they threw, lagged
- **passing_epa_lagged**: the Expected Points Added of a player based on their passing plays, lagged
- **passing_2pt_conversions_lagged**: the number of two point conversions a player gets via passes they threw, lagged
- **pacr_lagged**: Passing Air Conversion Ratio (passing yards/passing air yards) of a player, lagged
- **dakota_lagged**: DAKOTA score (adjusted EPA + CPOE weighted by coefficients that best predict the adjusted EPA/play in the previous year) for passing, lagged
- **carries_lagged**: the number of rush attempts by a player, lagged
- **rushing_yards_lagged**: the number of rushing yards by a player, lagged
- **rushing_tds_lagged**: the number of rushing touchdowns by a player, lagged
- **rushing_fumbles_lagged**: the number of fumbles a player has during rush attempts, lagged
- **rushing_fumbles_lost_lagged**: the number of fumbles a player *loses* during rush attempts, lagged
- **rushing_first_downs_lagged**: the number of first downs a player gets via a rush attempt, lagged
- **rushing_epa_lagged**: the Expected Points Added of a player based on their rushing plays, lagged
- **rushing_2pt_conversions_lagged**: the number of two point conversions a player gets via a rush attempt, lagged
- **receptions_lagged**: the number of catches by a player, lagged
- **targets_lagged**: the number of times a player was the intended receiver of a pass play, lagged
- **receiving_yards_lagged**: the number of receiving yards by a player, lagged
- **receiving_tds_lagged**: the number of receiving touchdowns by a player, lagged
- **receiving_fumbles_lagged**: the number of fumbles a player has after making a reception, lagged
- **receiving_fumbles_lost_lagged**: the number of fumbles a player *loses* after making a reception, lagged
- **receiving_air_yards_lagged**: the number of yards a receiving player receives solely from the pass distance (yards gained prior to any additonal yards gained after securing the reception), lagged
- **receiving_yards_after_catch_lagged**: the number of yards gained by a player after securing the reception, lagged
- **receiving_first_downs_lagged**: the number of first downs a player gets on plays where they made a reception, lagged
- **receiving_epa_lagged**: the Expected Points Added of a player based on their receiving plays, lagged
- **receiving_2pt_conversions_lagged**: the number of two point conversions a player gets via a reception they made, lagged
- **racr_lagged**: Receiver Air Conversion Ratio (receiving yards/receiving air yards), lagged
- **target_share_lagged**: the percentage of their team's total pass targets that a player receives, lagged
- **air_yards_share_lagged**: the percentage of their team's total air yards that a player receives, lagged
- **wopr_lagged**: Weighted Opportunity Rating (metric combining a player's target share and air yards) for receiving, lagged
- **special_teams_tds_lagged**: the number of touchdowns on special teams plays by a player, lagged
- **fantasy_points_ppr_lagged**: the number of PPR fantasy points a player has, lagged
### Lagged Stats Basis columns
- **used_opponent_avg**: indicator variable where 1 indicates the lagged stats are averaged from prior season games against the corresponding opponent_team
- **used_game_number_avg**: indicator variable where 1 indicates the lagged stats are from the corresponding prior season game_number
- **DNP_prior_season_game**: indicator variable where 1 indicates the player did not play in the prior season's game_number
- **never_seen**: indicator variable where 1 indicates the player has never been seen in the data before (i.e. rookies)

## Methodology
### Data Preparation/Cleaning
The dataframe used for modeling was built from 2 different datasets imported via nfl_data_py's import_weekly_data and import_schedules functions. import_weekly_data brought in weekly player stats, with each row corresponding to a player's stats in a given week and season in which they played. import_schedules brought in data on every matchup each week, such as the home team, away team, time, location, and more, which were not available in the weekly player stats data. These two dataframes were joined by season and week in addition to creating a "matchup" column in each that helped map each row to which teams were playing as the original schedules data contained a home and away team while the original weekly player stats data contained the team of the player and their opponent team. Afterwards, the game_number column was created in order to represent the specific game number of the season the player was playing, as bye weeks can distort the meaning behind week numbers in the dataframe. The data was also limited to just those weeks where traditional fantasy seasons were being played (up to week 16 through 2020 and up to week 17 afterwards) and to players in fantasy positions aside from kickers (and D/ST of course).
### Feature Engineering
In order to see how well models could predict a player's fantasy points in a week using their stats from the prior season, some tailored lagging of the weekly player stats had to be done. All the player stats columns were lagged to correspond to the player's stats in the prior season in 2 ways: their stats from the prior season's matchups with a team or their stats from the prior season's corresponding game number. As NFL teams play other teams in their division twice a year, composing of about half of their season's games each year, taking the average of these prior season matchups as the lagged stats for matchups in the current season was the primary way to lag stats. If a team also happened to play a team in back to back years just once, their stats from that matchup the prior season were used as the lagged stats for the current season as well. After getting the lagged stats for these matchups, the players lagged stats were then matched to the current season by the stats of their corresponding prior season's game number as bye weeks leave some players without stats in a prior season's week. Let's say we're creating lagged stats for Todd Gurley's 2018 campaign; here's an illustration of the process:
- Keep all identifier columns in 2018 as well as fantasy_points_ppr; this shows us his weekly fantasy points in 2018 as well as matchup data for each week
- Take the rest of his stats and lag them:
    - For games against teams he played the prior year, such as the Seattle Seahawks (a division rival), he would have stats from the prior year for 2 games. We take the average of his stats in these games in 2017 as the lagged stats for his games against them in this 2018 season
        - If in 2017 he rushed for 100 yards and 2 touchdowns in one and rushed for 150 yards, 1 rushing touchdown, and caught a 20 yard receiving touchdown in another, his stats for the 2018 matchups against the Seahawks would be the same for each matchup: 125 rushing yards, 1.5 rushing touchdowns, 10 receiving yards, 0.5 receiving touchdowns (an average day for him in both senses of the word back then)
    - Now let's say he played the Philadelphia Eagles once in 2017 and is scheduled to play them again in 2018; we would simply take his stats from this one matchup in 2017 as his lagged stats for the 2018 matchup
    - For all other games where he's playing against a team that he did not play in the prior season, such as if he were playing the Buffalo Bills in game number 6 of 2018 while not having had played them in 2017, we would then take his stats from 2017 game number 6 as the lagged stats for 2018 game number 6
- *Now remember*: while we're lagging all these stats accordingly (including fantasy points themselves), we are keeping their actual fantasy points each week as well as this is our target variable that should not be lagged.
    - By keeping this not lagged, we are having our models learn any patterns/interactions/relationships between fantasy points in the current year and stats from the prior year, which is our goal.

On top of this, binary indicator variables to show how if the data was lagged from prior matchups or from the prior game number were created in order to provide some distinction for how the stats data was created, as prior matchup data tended to be more consistent due to multiple division matchups each year while prior game number data could show stats from a game in which a player just so happened to have an off game the prior year in that same game number. Additionally, a binary indicator column telling if a player played in the prior season's game number or not was created to help further distinguish a player's stats as potentially misleading. As for the case of rookies or players who have never touched the field before (and thus cannot have any lagged stats to utilize), a binary indicator column was also created to distinguish this. 
### Limitations/Assumptions
When trying to predict any football statistics, much less fantasy football points, there are countless factors to consider to truly have the best shot at accurate predictions. As this project was simply out of curiosity to see how well lagged stats could do as features for predicting fantasy football points while also acting as an exercise for my own learning and practice creating machine and deep learning models, many factors that likely would have helped were left out. Some of these are the following:
- Opponent defensive data
    - Defensive ranks (against positions, rushing yards/touchdowns allowed, receiving yards/touchdowns allowed, scoring overall, yardage overall, etc.)
    - Defensive roster data (number of all-pros at each position, matchup data, etc.)
- Coaching schematics of both the player's team's offense and their opponent's defense
- Player personal statistics (not game statistics)
    - Height/weight
    - Age
    - Years under a coach/with other players
    - Major injury history
- Roster data of the player's team
    - Offensive ranks (by position, rushing yards/touchdowns per game, receiving yards/touchdowns per game, scoring overall, yardage overall, offensive line, etc.)
    - Stats of other players on their team
    - Injuries to players on their team
- Game script
    - Betting lines (moneyline, spread, handles)
    - Referees involved
    - Weather (wind, rain, snow, temperature, etc.)

While data on some of these isn't difficult to collect and use, others are going to be nearly impossible. For example, all head coaches of the 2025 season are known by now and can easily be mapped to each team to give insight into which coach players are playing under and playing against each week; however, data on coaching stats/schemes/playbooks would take a bit more digging. As for things like betting lines and weather, though available historically and easily attainable through nfl_data_py, there's almost no way to know ahead of time what these features' values would be for each game in the 2025 season, and thus they're rendered useless for the most part. Such hurdles were another reason I decided to limit the scope of this project to simply lagged stats, instead assuming that the results would be severely limited as well. Correlation tests of the features during EDA confirmed these suspicions as the highest correlation score with the target variable seen across all features in any context (split out into position groups and stats most related to that group) was only 0.38 and was the PPR fantasy points lagged themselves.
### Modeling
The 2025 weekly fantasy football points predictions were made via a CatBoost model and Multilayer Perceptron model. 

CatBoost is a gradient-boosting modeling method built to handle categorical features. It automatically pre-processes categorical features that are specified via target encoding, or calculating target variable statistics (i.e. mean) for each category. It additionally calculates these using an ordered subset of the data in order to target encode each row based on previous rows' values. This aids in avoiding target leakage as each row is target encoded without the knowledge of what that row's target variable is. As this dataset uses a number of categorical features that are important to factor in, CatBoost is naturally a good fit for ML modeling. CatBoost's ordered boosting (constructing training sets for each tree to prevent target leakage and reduce overfitting) and other regularization techniques also make it a great model choice for small to medium size datasets, such as the ~80,000 row dataframe used in this project. 

Multilayer Perceptrons (MLPs) are feedforward neural networks capable of learning complex, non-linear relationships by combining categorical and numerical inputs. Categorical features are passed through embedding layers to learn dense representations (vectors representing similarity patterns between categories), while numerical features are fed directly into the network. This makes MLPs a strong fit for datasets with mixed feature types. The ability to learn interactions between features through stacked dense layers enables MLPs to capture subtle patterns that simpler models may miss. Regularization and early stopping also make MLPs effective for small to medium-sized datasets, making an MLP a well-suited choice for this dataframe considering its wide-ranging categorical features and large number of features overall.

While models for specific position groups were created with CatBoost, their testing results were overall weaker than a model using all players, likely due to the features not having strong correlations regardless and the model simply having much more data to train upon with all players. As such, only one MLP that included all players was created, and only the CatBoost model that included all players and the sole MLP were used to generate predictions.

## Results
Results on the model's test sets were understandably (see [Limitations/Assumptions](#limitationsassumptions)) not great. The CatBoost model used (all players included) had an $R^2$ of only 0.32 and MAE of 4.8 on the test set; the best performing CatBoost model for a specific position (quarterbacks) was an $R^2$ of 0.31 and MAE of 5.9 on the test set. Even trials of taking out potential outliers, such as players who likely did not see the field much or suffered injuries/came in for injured players and thus had lower than the 25th percentile in scoring, resulted in worse scores, pointing to this most likely being simply a feature issue. The MLP did not perform any better, with a test MAE of 4.9. 

Overfitting was not much of an issue as evidenced by the CatBoost models often performing better on the test set than training or validation. While the MLP train MAE (4.5) was slightly better than the test MAE, hyperparameter tuning did not yield any improvements better than what I was able to achieve regardless of different combinations of dense layers and their L1 & L2 regularization that were tested. 
### SHAP Tests
SHAP tests were also run for CatBoost models in order to see which features were the most powerful predictors on all CatBoost models created. The results of these were intuitive: receiver predictions were most influenced by targets/target share, running backs by carries and other lagged rushing stats, quarterbacks by rushing yards and carries (rushing stats are worth more than passing stats), etc. The most interesting callout from SHAP tests was the quarterback model having more identifier features as its most impactful features, such as opponent_team and location showing up for this position group model but none of the others (including the all positions model). However, this also makes some intuitive sense as being the position offenses are run through and most frequently based around, quarterbacks are most frequently moreso affected by opponents' defensive schemes as well as where they play. 
### Top 10s
The top 10 players of each position group in terms of the 2025 season total predicted PPR fantasy points were extracted from the weekly predictions dataframe as the most interpretable takeaway from these predictions. Despite the models expectedly underperforming, the names in the top 10s were fairly sound. The most stark incorrect predictions were Rashee Rice being WR1 for the MLP, Carson Wentz being QB10 for the MLP, and Erick All being TE5 for the MLP; other more unreasonable surprises (in my opinion) were David Montgomery, James Conner, and Kareem Hunt being RBs 4-6 (above Saquon Barkely and Derrick Henry at that) for CatBoost, Deebo Samuel being WR9 for the MLP, and Aaron Rodgers being a top 10 QB for the MLP. Another callout of the models' underperformance should also be the trend of underpredicting season totals for most players in CatBoost: QB1 (Lamar Jackson) only scores 321 points, WR1 (Ja'Marr Chase) only scores 262, and only 1 TE breaks 200 points to name a few examples. The MLP results saw less underprediction overall, but instead saw some interesting overpredictions: RBs 1 & 2 (Christian McCafferey & Bijan Robinson) both scored about what Saquon Barkley did in his MVP-caliber 2024 season with RB3 Jahmyr Gibbs not far behind, the top 4 WRs all being around or above the 340 mark, and all top 10 TEs scoring north of 190 (the top 3 all scoring around 250).
### Gemini Analysis
With generative AI taking over our world more each passing day, I'm naturally curious how well its fantasy football knowledge might stack up to my own (shameless brag: I won my 12 person league in 2024). While LLM leaderboard-leading models unfortunately *definitely* know way more than me, in the context of this project, I was curious what some free models might have to say about my CatBoost and MLP's top 10 predictions. For this reason as well as my own practice and learning again, I created a function that RAG-backs (in a custom, tabular-data-driven way) Gemini's free Flash 2.0 model and generates analysis of a player's 2025 season total fantasy football points prediction based on a prompt I wrote, weekly predictions & their inputs, and the training data used in modeling.

Despite the extensive prompt I created involving persona-based, structure-based, and one-shot prompting, Flash 2.0 still ranked most of my predictions as decent (in the 65-85 range) despite some of the predictions being *very* questionable. Maybe I would have a fighting chance in a league against AIs; at least, as long as they're the lightweight free models.

## Next Steps
The purpose of this project was to find out how useful lagged stats alone could be in predicting fantasy football season totals of players as a way to practice my own ML/DL modeling and AI engineering. While it achieved this purpose, some next steps for a future iteration of it could be finding ways to utilize some of the other features I described in the [Limitations/Assumptions](#limitationsassumptions) section, predicting stats for kickers (they're people too!), and experimenting with other LLMs for the analysis generation. If you actually made it to the end here, thanks for reading, I hope you enjoyed this project, and feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/kevin-garand/) to talk fantasy football! (Or AI/ML/DL/data science, I suppose)