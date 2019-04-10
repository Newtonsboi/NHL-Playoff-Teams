Hi!

There are 3 algorithms implemented for this project:
 - kMeans.py
 - LogisticRegression.py
 - DeepNet.py

The other .py files are helper functions I implemented. genData.py loads, generates, and extracts the features from the csv data files. plotter.py is a plotting function for all the algorithms.

When you open one of the 3 algorithm files, just hit run and the program will run! If you search "Parameters", you can find the location in the code where you can play around with the parameters.

If you like the code, follow me on GitHub: https://github.com/Newtonsboi to look at other projects I'm working on. (shameless self-promo)

Enjoy!

Newtonsboi

============

Introduction
============

The excitement level of millions of people rises each October as the NHL
season begins. 31 teams, each playing 82 games, with the top teams
qualifying to the Stanley Cup playoffs. The NHL is split into 2
conferences, East and West, with each conference further split into 2
Divisions. 8 teams from each conference qualify for the playoffs: the
top 3 teams in each division, plus 2 “wild card” teams from each
conference. These wild card teams are the ones that finish and in their
conference. Can we use previous years data to predict which teams should
be in the playoffs this year? This is the question we are trying to
answer. As of writing this report, the regular season has ended, so we
can compare our machine-learned predictions to reality.

Making the Data-set
===================

The first step is making the data-set. We will use historical data
starting from the salary cap era (2005). Even though the NHL has been in
existence since 1917, prior to 2005, there was no restriction on the
amount of spending on players. With the enforcement of a salary cap, the
playing field became more level. The data was collected from the
website: Hockey Reference. For each of the 391 teams, a 0 is assigned if
the team made the playoffs that season, and 1 otherwise. The data
consists of 33 features. As our data-set is not large and we have a lot
of features, we want to avoid over-fitting. To do so, we need to cut
down the number of features.

Feature Selection
-----------------

We want to be able to predict a playoff team without knowing the team’s
ranking, or factors that influence ranking, as that would make the task
trivial. So, we got rid of these features. These are:

-   Ranking and Games Played

-   Wins and Losses (which include overtime and shootouts)

-   Strength of Schedule (this looks at strength of opponents)

-   Simple Rating System (this determines how good a team is)

We also make sure we do not have repetitive features. For example, there
is a Goals Scored feature, but also Even Strength Goals Scored, as well
as Power Play Goals Scored, the summation of the latter two giving the
former. So we removed the former. The result is that we are able to
reduce our features from 33 to 19. We would like to further reduce our
features. We found that the Random Forest Classifier has a tool which is
able to identify the important features. We use this to identify the Top
3 features. A visualization is shown below:

As seen in Figure \[fig:feature\], the top 3 features are Even Strength
Goals Against (EVGA), Goalie Save Percentage (SV%), and Even Strength
Goals For(EVGF). These 3 features are the ones we will use going
forward. Note, that the code has the capability of choosing more than 3
features.

Using K-Means to Classify
=========================

In this section we use K-Means to see if we can distinguish between
playoff and non-playoff teams based on the 3 features selected (EVGA,
EVGF, and SV%); thus, our K value is 2. To assess the accuracy of the
clustered points, we will compare the assigned points to each cluster to
the actual targets for both training and validation sets. We then look
at how this year’s teams cluster using K-Means and comment on how
accurate that is.

Results and Discussion
----------------------

First, we show our training and validation set plots in the figure
below. We can see that after around 70 epochs, the validation set loss
starts to increase, so we stop the training their:

When we compared the accuracy of this model to the actual data of
previous years, we found the clustering algorithm was 67% accurate on
both training and validation sets. This may seem low, but an argument
can be made that it is reasonable. The accuracy numbers can be
interpreted as: using statistics on just the goals scored, goals
against, and goalie save percentage, we are able to determine if a team
is a playoff team 67% of the time. In the real-world, there are many
more factors that influence if a team makes the playoffs, including
luck; however, these models can be very difficult to formulate.

You might ask why not use more or less features, as opposed to just 3.
We tried this for different feature set sizes: 2, 4, 8, and all of them,
and found the training set accuracy for each feature was 27%, 21%, 20%,
32% respectively, which is considerably less than when the feature size
is 3.

Let’s look at how K-Means clustered this year’s teams, where red
identifies teams in playoffs, and blue otherwise. This is shown in
Figure \[fig:my\_label\] below. Some observations:

-   It identified 15 teams making the playoffs, this is one short of
    what, where 16 teams make the playoffs.

-   You can see the difference between the best teams in the league,
    like the Tampa Bay Lightning, and the worst, the Ottawa Senators.

-   Of the 15 predicted playoff teams, 4 (Vancouver, Minnesota, Anaheim
    and Phoenix) of them did not qualify in reality and all four are in
    the same area in the plot. The area they are in is where they have
    better than league-average goalkeeping, but much worse scoring.

-   Of the 16 predicted non-playoff teams, 5 of them actually made it:
    Toronto, Columbus, Vegas, Washington and San Jose. The outlier here
    is San Jose, who have one of the worse SV% in the league, but make
    up for that by being on the higher end of EVGF, and playing in the
    weakest division this year: the Pacific.

-   With 9 incorrect predictions, the accuracy of the model on this
    year’s data is 71%, which is close to the training and validation
    accuracy.

Using Logistic Regression
=========================

In this section and the next, we factor into our training and validation
data whether a team made the playoffs, therefore we move from
unsupervised to supervised learning. The first supervised algorithm we
will use is Logistic. Our model is as follows: $$\begin{aligned}
    \sigma (x^TW+b)\end{aligned}$$ In this model, the outputted vector
is the probability of a team **not** making the playoffs. We use the
Adam Optimizer to find the weights and biases that minimize the model
above, and add a regularization term of 0.01 to avoid over-fitting the
data.

Results and Discussion
----------------------

The loss and accuracy plots on the training and validation data is shown
below:

The noise in the training set is likely due to the small batch size 13,
and the small training set size, 312 points. The final training and
validation set accuracy are 100% and 84.8%. Our final weight and bias
vectors are: $$\begin{aligned}
    W = \begin{bmatrix}
           -0.66,-1.67, 1.75
         \end{bmatrix} \text{,                 }
   b = \begin{bmatrix}
   -0.13
 \end{bmatrix}  \end{aligned}$$

Using these values we can calculate the probability of each team making
the playoffs. Unlike unsupervised learning were the algorithm only
divides teams into 0 or 1, the logistic model gives a range. The model
allows us to always choose the right number of playoff teams. Not only
can this model predict the playoff bound teams, but it can also predict
the first-round match-ups in the playoffs. For this year, the predicted
playoff match-ups are as follows: $$\begin{aligned}
    \textbf{Eastern Conference Match-ups:}\\
    \text{1. Tampa Bay Lightning } &\text{vs. } \text{WC2. Carolina Hurricanes}\\
    \text{2. Boston Bruins } &\text{vs. } \text{3. Toronto Maple Leafs}\\
    \text{1. New York Islanders } &\text{vs. } \text{WC1. Montreal Canadiens}\\
    \text{2. Pittsburgh Penguins } &\text{vs. } \text{3. Washington Capitals}\\
    \textbf{Western Conference Match-ups:}\\
    \text{1. Calgary Flames } &\text{vs. } \text{WC2. Colorado Avalanche}\\
    \text{2. Vegas Golden Knights } &\text{vs. } \text{3. San Jose Sharks}\\
    \text{1. Nashville Predators } &\text{vs. } \text{WC1. Winnipeg Jets}\\
    \text{2. Dallas Stars } &\text{vs. } \text{3. St. Louis Blues}\end{aligned}$$

Apart from the Canadiens, every team predicted to make the playoffs
matched reality! The Columbus Blue Jackets made it instead. So the
predictor got 2 teams wrong, yielding an accuracy of 92.3%. It might
seem like Montreal got really unlucky this year; looking at the figure
below, it seems the logistic predictor had them as the first Wild Card
team. Of note is how the model does well in segmenting the teams that
had no chance towards the end of the season in making the playoffs from
teams that were in the hunt for a spot; towards the end of the season,
the last 2 playoff spots was contested tightly between Montreal,
Carolina, and Columbus.

For predicted first round match-ups, only 3 of the match-ups were
predicted correctly: Boston vs Toronto, Calgary vs Colorado, and Vegas
vs San Jose. From these, 1 had the seeding incorrect: Vegas vs San Jose.

Using Neural Network to Classify 
================================

Now we look at a more complicated supervised learning algorithm: a
Neural Net. We implement a 3-layer MLP, with a drop-out layer between
the and the layer. The hidden unit layers each have 500 nodes. The
drop-out rate is set to 0.5, and just like in logistic regression, the
regularization is set to 0.01.

Results and Discussion 
----------------------

The loss and accuracy plots are shown below:

The final training and validation set accuracy are 100% and 83.5%. The
validation accuracy is 1% less than logistic regression, which is minor.
It seems like extending the classification of the data-set to a more
complicated algorithm did not make a difference. Using this neural net,
the predicted playoff match-ups are as follows: $$\begin{aligned}
    \textbf{Eastern Conference Match-ups:}\\
    \text{1. Tampa Bay Lightning } &\text{vs. } \text{WC2. Carolina Hurricanes}\\
    \text{2. Boston Bruins } &\text{vs. } \text{3. Toronto Maple Leafs}\\
    \text{1. New York Islanders } &\text{vs. } \text{WC1. Montreal Canadiens}\\
    \text{2. Pittsburgh Penguins } &\text{vs. } \text{3. Washington Capitals}\\
    \textbf{Western Conference Match-ups:}\\
    \text{1. Calgary Flames } &\text{vs. } \text{WC2. Colorado Avalanche}\\
    \text{2. Vegas Golden Knights } &\text{vs. } \text{3. San Jose Sharks}\\
    \text{1. Nashville Predators } &\text{vs. } \text{WC1. Dallas Stars}\\
    \text{2. St. Louis Blues } &\text{vs. } \text{3. Winnipeg Jets}\end{aligned}$$

Again, the model misclassifies Montreal and Columbus, yielding an
accuracy of 92.3%, the same as the simpler logistic model. In terms of
predicted first round match-ups, 5 of the match-ups were predicted
correctly: Boston vs Toronto, and the entire Western Conference. From
these, just 2 had the seeding incorrect: Vegas vs San Jose, and Winnipeg
vs St. Louis. Note that Winnipeg and St. Louis finished with the same
amount of points this year, but the tie-breaker for the higher seed went
to Winnipeg!

Conclusion
==========

In this project, our goal was to see if we can use data to analyze what
makes an NHL playoff team. We used 3 types of algorithms: K-Means,
Logistic, and a Neural Net. K-Means provided us with a graphical
intuition of what makes an NHL playoff team. This is true when our
feature set can be visualized in 3 or less dimensions. For example, for
this year, teams that made the playoffs were ones that had great
goalkeeping and did not let in a lot of goals; scoring a lot had less of
an impact. The down-side of the K-Means model though it does not take
advantage of the known targets (whether a team made the playoffs or not)
from the training data.

To take advantage of the known target values, we used logistic
regression and a neural network to classify the teams.It seems like the
added complexity of the neural net model only improves predicting the
first-round match-ups compared to the logistic model. In terms of
predicting the correct teams making the playoffs, both models returned
the same results. This makes sense: a neural nets advantage of being a
non-linear classifier is useful when we got a very large data-set. In
fact, the lack of data points (only 391) is the biggest critique of our
model. Machine Learning often realizes on large data-sets to train the
models. However, we believe we have done our best to counter-act this
issue; this is done by reducing the number of features from 19 to 3.
Nonetheless, having more data points would help solidify our models.
