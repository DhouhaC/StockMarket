
from grc1_toolbox.toolbox import GRCSubmission
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint

class Problem1Solution():

    def __init__(self):
        # self.lookback = 36  ## max number of historical datapoints you want at any given time
        self.lookback = -1 ## setting this to -1 will return everything till that point
        self.dataset = "G5"
        self.skip = 5 ## number of months you want to skip, for the first model

        self.params = {}

        # for example you can import and store an ML model from scikit learn in this dict
        self.model = {}

        # and set a frequency at which you want to update the model
        self.updateFrequency = 6

    #################################################################################################
    # Change this function to change your prediction model for predicting returns for the next time
    # period, don't use feature Y for training
    #
    # input argument data is a dataframe
    #################################################################################################
    def trainModel(self, data):
        y = data['Y']
        del data['Y']
        del data['Identifier']
        rforest_model = GradientBoostingClassifier(n_estimators=800, learning_rate=.15, max_depth=7,
                                                   min_samples_leaf=.05, random_state=0)

        rforest_model.fit(data, y)
        self.model = rforest_model

    #################################################################################################
    # This function predicts the next period returns for the assets given
    #
    #
    # input argument data_for_one_date is a dataframe, it doesn't have the feature Y
    #################################################################################################
    def getPrediction(self, data_for_one_date):
        data_for_one_date = data_for_one_date.drop(columns =['Identifier'])
        predictions = self.model.predict(data_for_one_date)

        ri = pd.Series(predictions, index=data_for_one_date.index)
        return ri


    #########################################################################################################
    # function to define constraints for optimization, this will give you a starting point
    # This doesn't have all the constraints implemented(turnover constraint) so,this optimizer won't adhere
    # to the constraints, also this might not be the best way to implement the optimizer, this is just to give you
    # a starting point
    #
    # if you need help in implementing these constraints look inside the toolbox files
    #
    # for more details see how to implement NonLinearConstraint in scipy
    #########################################################################################################

    def constraint(self, weights):
        wt = np.array(weights) # constraint for weights to be positive

        div_constraint = np.maximum(self.constraint_args['diversification'], 1 / float(self.Qt.sum()))
        cons2 = (wt/np.sum(wt))-div_constraint # diversification constraint

        cons4 = np.dot(weights, self.St) # spread constraint, these are also measure of risks for these assets

        cons5 = np.dot(weights, self.Dt) # duration constraint, these are also measure of risk for these assets

        return np.concatenate((wt,cons2, [ cons4, cons5]))

    #################################################################################################
    # Cost function for the optimizer, you will have to write your own cost function for the kind of
    # optimizer you are using, the way you write your cost function will determine a lot about your score
    #
    # for instance how do you decrease volatility
    #################################################################################################
    def cost(self, weights): # function to return the cost
        return -1*np.dot(weights, self.ri) + self.penalty*np.sum(np.abs(np.subtract(weights,self.w_initial.values)))




    #################################################################################################
    # change this function to return the weights, again you will have to write your logic depending
    # upon the kind of optimizer you are using
    #
    # the input arguments are
    # penalty : cost of transactions
    # identifiers : identifiers for the assets
    # wi : weight of the asset in the index
    # Dt : parameter for the asset
    # St : parameter for the asset
    # Qt : quantity
    # constraint_args : constraint arguments
    # df : dataframe with all other features for this date
    # trr : total return
    # wt_1 : last weights in the portfolio
    #################################################################################################
    def getWeights(self, penalty, identifiers, wi, Dt, St, Qt,constraint_args, df, trr, wt_1,
                   t11):
        self.penalty = penalty
        g = constraint_args['diversification']
        U = constraint_args['turnover']
        P = constraint_args['positions']
        delta = constraint_args['duration']
        chi = constraint_args['spread']
        cost_threshold = 0.00005


        self.constraint_args = constraint_args

        ri = self.getPrediction(df)

        self.Qt = Qt

        w_ini = 0 * Dt # initial weights
        if not (wt_1 is None):
            w_ini[w_ini.index.isin(wt_1.index)] = wt_1[wt_1.index.isin(w_ini.index)]

        # turnover allowed for this date
        turnover_allowed = U / 10 if wt_1 is None else np.maximum(
            U - t11 - np.sum(wt_1[~wt_1.index.isin(Qt[Qt == 1].index)].abs()), 0)

        self.w_initial = w_ini
        self.ri = ri[ri.index.isin(w_ini.index)]
        self.Dt = Dt[Dt.index.isin(w_ini.index)]
        self.St = St[St.index.isin(w_ini.index)]

        self.turnover = turnover_allowed

        div_constraint = np.maximum(g, 1 / float(self.Qt.sum()))

        ## setting lower and upper bounds for duration and spread
        duration_ub = np.dot(wi,Dt)*(1+delta)
        duration_lb = 0
        spread_ub =np.dot(wi, St) * (1 + chi)
        spread_lb = 0

        ## lower and upper bounds for all constraints
        lb = np.concatenate((np.zeros(len(w_ini)), -div_constraint * np.ones(len(w_ini)), [ spread_lb, duration_lb]), axis=0)
        ub = np.concatenate((np.ones(len(w_ini)), 0.001 * np.ones(len(w_ini)), [ spread_ub, duration_ub]))

        ## intialising the constraints
        constr = NonlinearConstraint(self.constraint, lb=lb, ub=ub)
        result = minimize(self.cost, constraints=(constr) , x0= np.ones(len(w_ini)))

        print("Optimizer status : ", result.success)
        wt = ri
        wt[wt.index.isin(w_ini.index)] = pd.Series(result.x, index=w_ini.index)
        wt[~wt.index.isin(w_ini.index)] = 0
        wt = wt/wt.sum()
        wt[wt.abs()<1e-8]=0
        return wt,result.fun


if __name__ == "__main__":
    p1 = Problem1Solution()

    sub = GRCSubmission(p1)
    print('Loaded Problem Params, Loading Backtester and Data')
    sub.startTrading()
    metrics = sub.getMetrics()
    dates = sorted(list(metrics.keys()))
    print("Your score based on your features is %s" % metrics[dates[-1]]['sortino Ratio'])
