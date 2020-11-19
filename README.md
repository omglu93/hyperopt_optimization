# Online shopping intentions - Classification problem & Hyperopt optimization

The goal of this notebook is to showcase the Bayesian (SMBO) hyperparameter optimization technique on a simple classification problem. The library that I am going to use for this endeavour is Hyperopt, which I can only recommend for its simple usage and excellent results in fine-tuning the model.


![los](/images/shopping.jpg)


# Primary goal : Create a LightGBM ML model for the classification problem 

The classification problem was tackled with LightGBM, a tree-based machine-learning algorithm that dominated multiple Kaggle competitions. It is generally a great choice due to its many benefits regarding the speed and accuracy of predictions. Also, its GOSS and EFB inner workings allow me to play around with data on a decade-old laptop.


# Secondary goal : Hyperparameter optimization using the Bayesian method

Hyperparameter optimization is generally one of the more time-consuming and costly tasks.  The Bayesian method alleviates some those issues in comparison to other available strategies. It focuses is more on the selection process of the hyperparameters that might yield improvement to the model rather than brute-forcing or random selection. The general idea behind it is that the algorithm proposes a set of hyperparameter candidates and evaluates them using the actual objective function. Those results are stored along with their respective candidates and used to construct/improve a probability model of the objective function. Repeating the evaluation process improves the probability function with every iteration and subsequently uses the probability function to select the hyperparameters with the greatest "Expected improvement".

The algorithm spends some more time with the selection process of the next hyperparameter to maximize the "Expected improvement" compared to the alternatives. However, it is still much cheaper in computational cost by spending less time evaluating poor hyperparameter choices.

# Dataset

The data is provided by Yomi Kastro & C. Okan Sakar.

- 18 features/columns
  - 10 numerical & 8 categorical
- 12,350 rows


# Requirements
- Python 2.7 or Python 3.6
- Jupyter Notebook

# License
MIT. See the LICENSE file for the copyright notice.

# References:

1. https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
2. https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf
