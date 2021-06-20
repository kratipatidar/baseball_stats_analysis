## Analysis of Baseball Statistics

### Abstract
Baseball is one of the most popular games in the US. So naturally, it has a plethora of statistics associated with it. There are various batting, pitching and fielding statistics, that can help determine the result of a game to a high degree of certainty.

The aim of this project is to utilize the baseball dataset, and come up with new features that would help in predicting whether the home team wins a particular match or not.

For achieving this, first a good amount of feature engineering was carried out on the data, to come up with predictors. Next, different kinds of plots were obtained to gain an in-depth understanding of how the predictor interacts with the response. Added to this, the interaction between different predictors was also observed through these plots to help segregate predictors that add value to our prediction analysis from predictors that hamper our predictions.

After this, several classification models were fit on the data, and accuracy scores were obtained on these fits. A train-test split was done to test the model performance on unseen data.

Lastly, all these steps were wrapped into a docker container, which made up the production phase of this project.

### Dataset and Feature Creation
This dataset contains information on the different baseball statistics for all the MLB games that occurred between 2007 and 2012.

For the purpose of our analysis, we will consider the following statistics to help model the relationship between the predictors and the response.


| Statistic| Description|
|----------|------------|
|At bats per home run|at bats divided by home runs|
|Batting Average|hits divided by at bats (H/AB)|             
|Walk to Strikeout Ratio|number of bases on balls divided by number of strikeouts|                    
|Batting average on balls in play| frequency at which a batter reaches a base after putting the ball in the field of play|                                      
|Ground ball fly ball ratio|number of ground ball outs divided by number of fly ball outs|            
|Gross production average|1.8 times on-base percentage plus slugging percentage, divided by four|
|Hit by pitch|times touched by a pitch and awarded first base as a result|
|Home runs per hit|home runs divided by total hits|
|Isolated power|a hitter's ability to hit for extra bases, calculated by subtracting batting average from slugging percentage|
|On-base percentage|times reached base (H + BB + HBP) divided by at bats plus walks plus hit by pitch plus sacrifice flies (AB +BB +HBP +SF)|
|On-base plus slugging|on-base percentage plus slugging average|
|Plate appearances per strikeout|number of times a batter strikes out to their plate appearance|
|Slugging average| total bases achieved on hits divided by at-bats (TB/AB)|
|Total bases| one for each single, two for each double, three for each triple, and four for each home run|
|Times on base| times reaching base as a result of hits, walks, and hit-by-pitches (H + BB + HBP)|
| Extra base hits| total hits greater than singles (2B + 3B + HR)|

Source : https://en.wikipedia.org/wiki/Baseball_statistics

### Brute Force Variable Combination
The difference with mean of response for each combination of two predictors yielded the brute force variable combination plots as illustrated below.

![image](https://user-images.githubusercontent.com/55113076/102708830-5b938980-4273-11eb-8f8e-10201a0c9bcb.png)

![image](https://user-images.githubusercontent.com/55113076/102722713-f5693e00-4328-11eb-840e-a72b8e955ccd.png)

Around 1089 plots were obtained. However, there isn't any significant relationship that we observe from these plots.

Thus, we proceed with further analysis by using our existing predictors.

### Correlation between Predictors
When I first started off with my analysis, I created all my predictors separately for home and away teams. The correlation plot in such a predictor setting is illustrated below.

![image](https://user-images.githubusercontent.com/55113076/102722807-a8d23280-4329-11eb-9265-136e65d5e3b7.png)


Since there was a lot of correlation between predictors such as XBH, TB and TOB, I proceeded to do a difference and ratio between the different predictors.

However, this did not entirely eliminate the correlation and there still remained some correlated predictors. 

![image](https://user-images.githubusercontent.com/55113076/102709105-3b64ca00-4275-11eb-87eb-c51ea4dd6dc9.png)


The yellow here represents some highly correlated predictors, with a correlation coefficient of close to 1.

### Difference with Mean of Response
The difference with mean of response plots obtained for several of the predictors are illustrated below.

![image](https://user-images.githubusercontent.com/55113076/102722922-8a206b80-432a-11eb-9e78-9622e51abd2f.png)

![image](https://user-images.githubusercontent.com/55113076/102722929-999fb480-432a-11eb-8699-20ae12f6ed0b.png)

![image](https://user-images.githubusercontent.com/55113076/102722987-f8fdc480-432a-11eb-8d36-6e7fdeb1d841.png)

### Interaction between the Response and Predictors
I plotted different predictors versus the response. I saw a lot of overlap between the distributions. Here is an example of such a plot.

![image](https://user-images.githubusercontent.com/55113076/102722601-239a4e00-4328-11eb-9ab2-931e685b325f.png)

The other alternative to this is taking the difference and ratio of these predictors, and observing if these predictor plots offer any improvement over the previously obtained plots.

In my analysis, I observed that even the ratio and differences, for instance the SLG_home and SLG_diff plots had significant overlap between them.

![image](https://user-images.githubusercontent.com/55113076/102708897-d6f53b00-4273-11eb-9966-14ab1ed64226.png)

This plot clearly shows the overlap for predictors with a 0 and a 1 response respectively, for the Slugging Average Difference Predictor.

### Model Building and Performance
Namely, 4 classification models were employed for determining the performance of the result on test set. An 80-20 train - test split was done for this purpose. 

|Model| Performance |
|-----|----------|
|Logistic Regression|0.539491|
|SVM Classifier|0.532963|
|Random Forest Classifier|0.53329|
|Naive Bayes Classifier|0.487598|

There was a slight improvement in performance in this case, as compared to the case where separate predictors were created for each of the away and home teams.

### Conclusion and Future Scope
I believe I spent a great deal of time on understanding the dataset and various baseball metrics, and devoted time to just manipulating a specific set of baseball metrics, i.e., the batting statistics.

Using the pitching and fielding statistics is one way of adding information to the models and improving prediction results.

Another way would be getting rid of or modifying the predictors that hamper the analysis results, and do not add value to the model performance.

