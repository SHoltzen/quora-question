# What am I looking at? #
This is the repository for my solution to the quora question: Given certain data about a question, can you predict whether or not it will be answered within a single day? 

The original challenge posting can be found at (hackerrank)[https://www.hackerrank.com/contests/quora/challenges/quora-ml-answered]. 

The format for the challenge is to read in from standard in answered_data_10k.in and write data in the format of answered_data_10k.out. The more closely
your output matches answered_data_10k.out, the better your solution. 

I created this solution in ~6 hours of programming (it was a timed competition).

This project was a very interesting learning experience for me. My job as Symantec has taught me many things about machine learning and how to approach problems such as this,
but I have never had to create a classification scheme in 6 hours. What follows is a description of my thought process, along with my mistakes and successes.

# How I approached this problem #

My first step is to *analyze the attributes that are readily available from the data set. My weapon of choice when it comes to these sorts of problems is 
(weka)[http://www.cs.waikato.ac.nz/ml/weka/]. It's a very easy to use (but unfortunately not very performant) machine learning library that makes it extremely
easy to test and compare classifiers, as well as examine attributes. 

Thus my first step was to generate numeric attributes that could be used in Weka, and output them to a format that Weka can understand ((ARFF)[http://www.cs.waikato.ac.nz/ml/weka/arff.html]).

Here are some of the simple topics that I initially considered:

1. Total follows: A simple tally of how many total people are following a question in all its topics.

2. Anonymous: Whether or not the question was posed anonymously.

3. Number of topics: a simple tally of how many topics a question has.

4. Most popular topic: The number of followers in the most popular topic.

5. Context topic followers: How many followers that the most important topic to a question has.

6. Question length: The length of the question text. 

I evaluated these topics by creating a C4.5 decision tree in Weka, which generated the following output:
```
Attributes:   7
              totalfollows
              anonymous
              numtopics
              mostpopulartopic
              contexttopicfollowers
              questionlength
              class
Test mode:10-fold cross-validation

=== Classifier model (full training set) ===

J48 pruned tree
------------------

numtopics <= 1: false (2943.0/975.0)
numtopics > 1
|   totalfollows <= 1830
|   |   numtopics <= 3: false (288.0/91.0)
|   |   numtopics > 3
|   |   |   numtopics <= 4
|   |   |   |   anonymous = false: true (24.0/11.0)
|   |   |   |   anonymous = true
|   |   |   |   |   totalfollows <= 769: true (2.0)
|   |   |   |   |   totalfollows > 769
|   |   |   |   |   |   totalfollows <= 1239: false (3.0)
|   |   |   |   |   |   totalfollows > 1239: true (2.0)
|   |   |   numtopics > 4
|   |   |   |   mostpopulartopic <= 418: true (5.0)
|   |   |   |   mostpopulartopic > 418: false (9.0/2.0)
|   totalfollows > 1830
|   |   numtopics <= 3
|   |   |   contexttopicfollowers <= 295: false (504.0/212.0)
|   |   |   contexttopicfollowers > 295: true (2133.0/929.0)
|   |   numtopics > 3
|   |   |   questionlength <= 189: true (3041.0/1052.0)
|   |   |   questionlength > 189
|   |   |   |   contexttopicfollowers <= 2198: false (17.0/2.0)
|   |   |   |   contexttopicfollowers > 2198
|   |   |   |   |   numtopics <= 7
|   |   |   |   |   |   questionlength <= 208: true (11.0/2.0)
|   |   |   |   |   |   questionlength > 208
|   |   |   |   |   |   |   numtopics <= 5
|   |   |   |   |   |   |   |   questionlength <= 243
|   |   |   |   |   |   |   |   |   questionlength <= 218: false (2.0)
|   |   |   |   |   |   |   |   |   questionlength > 218: true (3.0)
|   |   |   |   |   |   |   |   questionlength > 243: false (3.0)
|   |   |   |   |   |   |   numtopics > 5: false (6.0)
|   |   |   |   |   numtopics > 7: true (4.0)

Number of Leaves  :     18

Size of the tree :  35


Time taken to build model: 0.2 seconds

=== Stratified cross-validation ===
=== Summary ===

Correctly Classified Instances        5579               61.9889 %
Incorrectly Classified Instances      3421               38.0111 %
Kappa statistic                          0.2395
Mean absolute error                      0.4632
Root mean squared error                  0.4846
Relative absolute error                 92.6502 %
Root relative squared error             96.928  %
Total Number of Instances             9000     

=== Detailed Accuracy By Class ===

               TP Rate   FP Rate   Precision   Recall  F-Measure   ROC Area  Class
                 0.55      0.31       0.638     0.55      0.591      0.639    false
                 0.69      0.45       0.606     0.69      0.645      0.639    true
Weighted Avg.    0.62      0.381      0.622     0.62      0.618      0.639
```
From this you can visualize the decision tree, as well as examine its performance. It is a very good strategy to examine a decision tree to see how 
attributes are used in concert. Even if a decision tree is not going to be your final classification agent, it is still an enlightening exercize and can 
reveal relationships between various attributes. Notice the following things:

* Cross fold validation: This means that I examined the performance of the tree against 10 random holdout sets, which eliminates potential selection errors.

* Num topics: If you have less than 1 topic, you're overwhelmingly likely to be unanswered. This is interesting.

* Context follower usage: Notice how the context follower attribute gets used. It matter most when there are few topics (less than 3).

* Potential over-training on question length. Notice it's being used twice in the same branch
```
|   |   |   |   |   |   |   |   questionlength <= 243
|   |   |   |   |   |   |   |   |   questionlength <= 218: false (2.0)
|   |   |   |   |   |   |   |   |   questionlength > 218: true (3.0)
|   |   |   |   |   |   |   |   questionlength > 243: false (3.0)
```
This is very unusual, and makes me believe that question length is not a useful predictive feature. Why would being greater than 218 and less than 243 in length be the 
difference between being answered and unanswered? It's just not logical. Sure enough, if I rerun the same classifier without question length, I get 62% accuracy.

Now, 61.98% is better than randomly guessing, but not much better. I was hoping for much better. So, time to sit down and actually think of a useful attribute.

# The Useful Attribute #
This is where machine learning gets interesting. After taking a break from working and coming back, I realize that the number of followers isn't very important:
far more important is the nature of those followers. It makes sense that some topics will have very active users that constantly answer questions, while other topics
have questions that languish in unanswering even if it has thousands of followers. This led me to come up with my most valuable feature, and the reason that I won the 
Quora competition: *question answering rate per topic*. 

The way that I computed this feature is by computing percent answer rate of each topic in the training set. Then, for each question, I found the max answer rate of 
any one topic, and used this as a feature. I chose the max answer rate instead of a sum total in order to make the attribute more resistant to missing topics (a sum
total heavily favors topics that are known, even if they are not popular).

Now I'm going to show you the results.
A quick disclaimer: In order to properly evaluate this feature,
I would have to calculate it on a hold out set. Time constraints made this very difficult, so I did not evaluate this feature against the holdout, but rather against the
entire data set. This means that the following statistics will be vastly inflated, but still informative. 

After training a C4.5 tree with this attribute, I got the following output from Weka:
```
Scheme:weka.classifiers.trees.J48 -C 0.25 -M 5
Relation:     output-weka.filters.unsupervised.attribute.Remove-R1-weka.filters.unsupervised.attribute.Remove-R7
Instances:    9000
Attributes:   7
              topicprob
              totalfollows
              anonymous
              numtopics
              mostpopulartopic
              contexttopicfollowers
              class
Test mode:10-fold cross-validation

=== Classifier model (full training set) ===

J48 pruned tree
------------------

topicprob <= 1.493523
|   numtopics <= 2
|   |   topicprob <= 0.999261
|   |   |   numtopics <= 1
|   |   |   |   topicprob <= 0.754209: false (2637.0/718.0)
|   |   |   |   topicprob > 0.754209: true (147.0/49.0)
|   |   |   numtopics > 1: false (527.0/38.0)
|   |   topicprob > 0.999261
|   |   |   numtopics <= 1: true (159.0)
|   |   |   numtopics > 1
|   |   |   |   topicprob <= 1.31638
|   |   |   |   |   anonymous = false
|   |   |   |   |   |   topicprob <= 1.264706: false (264.0/122.0)
|   |   |   |   |   |   topicprob > 1.264706: true (36.0/9.0)
|   |   |   |   |   anonymous = true: false (136.0/63.0)
|   |   |   |   topicprob > 1.31638: true (179.0/38.0)
|   numtopics > 2: false (805.0/34.0)
topicprob > 1.493523
|   numtopics <= 2: true (330.0/4.0)
|   numtopics > 2
|   |   topicprob <= 2.298469
|   |   |   numtopics <= 3
|   |   |   |   topicprob <= 1.87381
|   |   |   |   |   contexttopicfollowers <= 137371
|   |   |   |   |   |   anonymous = false: false (158.0/77.0)
|   |   |   |   |   |   anonymous = true
|   |   |   |   |   |   |   mostpopulartopic <= 7637: true (6.0)
|   |   |   |   |   |   |   mostpopulartopic > 7637: false (70.0/24.0)
|   |   |   |   |   contexttopicfollowers > 137371: true (71.0/21.0)
|   |   |   |   topicprob > 1.87381: true (348.0/46.0)
|   |   |   numtopics > 3
|   |   |   |   numtopics <= 4
|   |   |   |   |   topicprob <= 1.996359: false (130.0/18.0)
|   |   |   |   |   topicprob > 1.996359
|   |   |   |   |   |   anonymous = false: true (88.0/38.0)
|   |   |   |   |   |   anonymous = true: false (48.0/20.0)
|   |   |   |   numtopics > 4: false (190.0/4.0)
|   |   topicprob > 2.298469
|   |   |   numtopics <= 4
|   |   |   |   topicprob <= 2.695238
|   |   |   |   |   numtopics <= 3: true (237.0/1.0)
|   |   |   |   |   numtopics > 3
|   |   |   |   |   |   contexttopicfollowers <= 258252
|   |   |   |   |   |   |   totalfollows <= 302998
|   |   |   |   |   |   |   |   topicprob <= 2.515422
|   |   |   |   |   |   |   |   |   mostpopulartopic <= 67927: true (45.0/10.0)
|   |   |   |   |   |   |   |   |   mostpopulartopic > 67927
|   |   |   |   |   |   |   |   |   |   contexttopicfollowers <= 109074: false (18.0/5.0)
|   |   |   |   |   |   |   |   |   |   contexttopicfollowers > 109074: true (11.0/4.0)
|   |   |   |   |   |   |   |   topicprob > 2.515422: true (75.0/14.0)
|   |   |   |   |   |   |   totalfollows > 302998: true (84.0/8.0)
|   |   |   |   |   |   contexttopicfollowers > 258252: false (19.0/7.0)
|   |   |   |   topicprob > 2.695238: true (539.0/5.0)
|   |   |   numtopics > 4
|   |   |   |   topicprob <= 3.210753
|   |   |   |   |   numtopics <= 6
|   |   |   |   |   |   topicprob <= 2.813725: false (128.0/32.0)
|   |   |   |   |   |   topicprob > 2.813725
|   |   |   |   |   |   |   numtopics <= 5
|   |   |   |   |   |   |   |   mostpopulartopic <= 25299: true (11.0)
|   |   |   |   |   |   |   |   mostpopulartopic > 25299
|   |   |   |   |   |   |   |   |   anonymous = false: true (59.0/20.0)
|   |   |   |   |   |   |   |   |   anonymous = true
|   |   |   |   |   |   |   |   |   |   contexttopicfollowers <= 5099: false (7.0/1.0)
|   |   |   |   |   |   |   |   |   |   contexttopicfollowers > 5099
|   |   |   |   |   |   |   |   |   |   |   topicprob <= 3.076464
|   |   |   |   |   |   |   |   |   |   |   |   topicprob <= 2.976718: true (7.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   topicprob > 2.976718: false (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   topicprob > 3.076464: true (7.0)
|   |   |   |   |   |   |   numtopics > 5
|   |   |   |   |   |   |   |   mostpopulartopic <= 474047: true (27.0/11.0)
|   |   |   |   |   |   |   |   mostpopulartopic > 474047: false (10.0)
|   |   |   |   |   numtopics > 6: false (64.0)
|   |   |   |   topicprob > 3.210753
|   |   |   |   |   topicprob <= 5.618282
|   |   |   |   |   |   numtopics <= 7
|   |   |   |   |   |   |   topicprob <= 3.971334
|   |   |   |   |   |   |   |   numtopics <= 5: true (272.0/10.0)
|   |   |   |   |   |   |   |   numtopics > 5
|   |   |   |   |   |   |   |   |   numtopics <= 6
|   |   |   |   |   |   |   |   |   |   topicprob <= 3.74768
|   |   |   |   |   |   |   |   |   |   |   anonymous = false: true (48.0/20.0)
|   |   |   |   |   |   |   |   |   |   |   anonymous = true
|   |   |   |   |   |   |   |   |   |   |   |   totalfollows <= 172495: true (7.0)
|   |   |   |   |   |   |   |   |   |   |   |   totalfollows > 172495
|   |   |   |   |   |   |   |   |   |   |   |   |   totalfollows <= 345037: false (5.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   totalfollows > 345037
|   |   |   |   |   |   |   |   |   |   |   |   |   |   mostpopulartopic <= 258252: true (6.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   mostpopulartopic > 258252
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   contexttopicfollowers <= 29069: true (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   |   |   |   |   |   contexttopicfollowers > 29069: false (5.0/1.0)
|   |   |   |   |   |   |   |   |   |   topicprob > 3.74768: true (39.0/2.0)
|   |   |   |   |   |   |   |   |   numtopics > 6
|   |   |   |   |   |   |   |   |   |   topicprob <= 3.470085: false (12.0/1.0)
|   |   |   |   |   |   |   |   |   |   topicprob > 3.470085
|   |   |   |   |   |   |   |   |   |   |   contexttopicfollowers <= 113827: false (17.0/6.0)
|   |   |   |   |   |   |   |   |   |   |   contexttopicfollowers > 113827: true (10.0/1.0)
|   |   |   |   |   |   |   topicprob > 3.971334: true (470.0/9.0)
|   |   |   |   |   |   numtopics > 7
|   |   |   |   |   |   |   topicprob <= 4.259454: false (50.0/3.0)
|   |   |   |   |   |   |   topicprob > 4.259454
|   |   |   |   |   |   |   |   numtopics <= 9
|   |   |   |   |   |   |   |   |   numtopics <= 8: true (43.0/9.0)
|   |   |   |   |   |   |   |   |   numtopics > 8
|   |   |   |   |   |   |   |   |   |   contexttopicfollowers <= 5195: false (5.0)
|   |   |   |   |   |   |   |   |   |   contexttopicfollowers > 5195: true (9.0/3.0)
|   |   |   |   |   |   |   |   numtopics > 9: false (17.0/1.0)
|   |   |   |   |   topicprob > 5.618282: true (298.0/2.0)

Number of Leaves  :     55

Size of the tree :  109


Time taken to build model: 0.22 seconds

=== Stratified cross-validation ===
=== Summary ===

Correctly Classified Instances        7312               81.2444 %
Incorrectly Classified Instances      1688               18.7556 %
Kappa statistic                          0.625 
Mean absolute error                      0.2539
Root mean squared error                  0.3642
Relative absolute error                 50.7821 %
Root relative squared error             72.8465 %
Total Number of Instances             9000     

=== Detailed Accuracy By Class ===

               TP Rate   FP Rate   Precision   Recall  F-Measure   ROC Area  Class
                 0.876     0.251      0.776     0.876     0.823      0.883    false
                 0.749     0.124      0.859     0.749     0.8        0.883    true
Weighted Avg.    0.812     0.187      0.818     0.812     0.812      0.883

=== Confusion Matrix ===

    a    b   <-- classified as
 3932  555 |    a = false
 1133 3380 |    b = true

```
Correctly classified instance count of 81%?! That's what I like to see. This tree is significantly more complex than the previous tree, and thus provides
much less insight into how particular attributes are valued. However, it is clear that topicprob is the most valuable attribute by far.

Once I determined my attributes I began to implement my solution.

# Implementing the Solution #
If you looked at my solution, it is noticeably different from what I have outlined here. It appears to have two classifiers trained on different data,
as well as 
My solution is written in Java so that I can use Weka. Once I had tested it and ensured that it worked, I submitted it to hacker rank and was horrified by the results.
I got a mere 61% accuracy. Top 10, but not the best. So why did this happen?

It turns out many of the questions in the hidden evaluation set contained topics that were not present in the training set. This combined with my earlier disclaimer
meant that my solution failed miserably on the hidden data set.

In order to improve this, I created a 2-classifier scheme: one for a question with all known topics, and another for when no known topics could be found. This
achieved 64% accuracy on the challenge, which was barely enough. 

I also tuned changed my classifiers. Through trial and error, I determined that a Bayes-Net classifier was the right combination of generalizability and 
accuracy on the training set to get the highest performance.



