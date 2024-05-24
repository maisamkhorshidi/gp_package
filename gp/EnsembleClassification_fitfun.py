import numpy as np
from sklearn.neural_network import MLPClassifier
from .utils import sigmoid, argmax, logloss
import re

def EnsembleClassification_fitfun(evalstr1, gp):
    ensemble_method = gp['runcontrol']['ensemble_method']
    ytr = gp['userdata']['ytrain']
    yval = gp['userdata']['yvalid']
    yts = gp['userdata']['ytest']
    ybinarytr = gp['userdata']['ybinarytrain']
    ybinaryval = gp['userdata']['ybinaryvalid']
    ybinaryts = gp['userdata']['ybinarytest']
    numDatatr = ytr.shape[0]
    numDataval = yval.shape[0]
    numDatats = yts.shape[0]
    popNum = gp['runcontrol']['num_pop']
    clsNum = gp['runcontrol']['num_class']
    initial_learning_rate = gp['runcontrol']['initial_learning_rate']
    lambda_L1 = gp['runcontrol']['lambda_L1']
    lambda_L2 = gp['runcontrol']['lambda_L2']
    tolgrad = gp['runcontrol']['tolgrad']
    epsilon = gp['runcontrol']['epsilon']
    max_iteration = gp['runcontrol']['max_iteration']
    sumnumGenes = np.zeros(popNum)
    numGenes = [np.zeros(len(evalstr1[jj])) for jj in range(popNum)]

    for jj in range(popNum):
        for kk in range(1):
            numGenes[jj][kk] = len(evalstr1[jj][kk])
        sumnumGenes[jj] = sum(numGenes[jj])

    weightsindiv = np.zeros(popNum)
    fitnesstrain = []
    fitnessvalid = []
    fitnesstest = []
    weightsgene = [None] * popNum
    gradient = [None] * popNum
    yptrain = [None] * popNum
    ypvalid = [None] * popNum
    yptest = [None] * popNum
    probtrain = [None] * popNum
    probvalid = [None] * popNum
    probtest = [None] * popNum
    losstrain = np.zeros(popNum)
    lossvalid = np.zeros(popNum)
    losstest = np.zeros(popNum)

    pat = r'x(\d+)'
    evalstrtr = [None] * len(evalstr1)
    evalstrval = [None] * len(evalstr1)
    evalstrts = [None] * len(evalstr1)

    for jj in range(len(evalstr1)):
        evalstrtr[jj] = [None] * len(evalstr1[jj])
        evalstrval[jj] = [None] * len(evalstr1[jj])
        evalstrts[jj] = [None] * len(evalstr1[jj])
        for kk in range(len(evalstr1[jj])):
            evalstrtr[jj][kk] = [re.sub(pat, r'gp["userdata"]["xtrain"][:,\1]', expr) for expr in evalstr1[jj][kk]]
            evalstrval[jj][kk] = [re.sub(pat, r'gp["userdata"]["xvalid"][:,\1]', expr) for expr in evalstr1[jj][kk]]
            evalstrts[jj][kk] = [re.sub(pat, r'gp["userdata"]["xtest"][:,\1]', expr) for expr in evalstr1[jj][kk]]

    geneOutputstr = [None] * popNum
    geneOutputsval = [None] * popNum
    geneOutputsts = [None] * popNum

    for jj in range(popNum):
        geneOutputstr[jj] = np.zeros((numDatatr, int(sumnumGenes[jj])))
        geneOutputsval[jj] = np.zeros((numDataval, int(sumnumGenes[jj])))
        geneOutputsts[jj] = np.zeros((numDatats, int(sumnumGenes[jj])))
        nn = 1
        for kk in range(1):
            for gg in range(int(numGenes[jj][kk])):
                exec(f'geneOutputstr[jj][:, nn] = sigmoid({evalstrtr[jj][kk][gg]})')
                exec(f'geneOutputsval[jj][:, nn] = sigmoid({evalstrval[jj][kk][gg]})')
                exec(f'geneOutputsts[jj][:, nn] = sigmoid({evalstrts[jj][kk][gg]})')

                if not np.isfinite(geneOutputstr[jj][:, nn]).all() or not np.isreal(geneOutputstr[jj][:, nn]).all():
                    gp['fitness']['returnvalues'][gp['state']['current_individual']] = []
                    fitness = np.inf
                    return fitness, gp
                nn += 1

    genetr = [None] * popNum
    geneval = [None] * popNum
    genets = [None] * popNum

    for jj in range(popNum):
        subset = np.eye(int(sumnumGenes[jj]))
        genetr[jj] = np.zeros((numDatatr, subset.shape[0]))
        geneval[jj] = np.zeros((numDataval, subset.shape[0]))
        genets[jj] = np.zeros((numDatats, subset.shape[0]))
        for cc in range(subset.shape[0]):
            genetr[jj][:, cc] = np.prod(geneOutputstr[jj][:, subset[cc, :] == 1], axis=1)
            geneval[jj][:, cc] = np.prod(geneOutputsval[jj][:, subset[cc, :] == 1], axis=1)
            genets[jj][:, cc] = np.prod(geneOutputsts[jj][:, subset[cc, :] == 1], axis=1)

    geneOutputstr1 = [None] * popNum
    geneOutputsval1 = [None] * popNum
    geneOutputsts1 = [None] * popNum

    for jj in range(popNum):
        geneOutputstr1[jj] = [None] * 1
        geneOutputsval1[jj] = [None] * 1
        geneOutputsts1[jj] = [None] * 1
        for kk in range(1):
            geneOutputstr1[jj][kk] = genetr[jj]
            geneOutputsval1[jj][kk] = geneval[jj]
            geneOutputsts1[jj][kk] = genets[jj]

    if not gp['state']['run_completed'] or gp['state']['force_compute_theta']:
        net = [None] * popNum
        for cc in range(popNum):
            geneOutputstr2 = geneOutputstr1[cc]
            geneOutputsval2 = geneOutputsval1[cc]
            try:
                Mdl = MLPClassifier(hidden_layer_sizes=gp['runcontrol']['layer'])
                Mdl.fit(geneOutputstr2[0], ytr)
                net[cc] = Mdl
                weightsgene[cc] = Mdl.coefs_
                gradient[cc] = Mdl.loss_curve_
                lossvalid[cc] = Mdl.loss_
            except:
                losstrain[cc] = np.inf
                lossvalid[cc] = np.inf
                losstest[cc] = np.inf

        if np.isinf(losstrain).any():
            gp['fitness']['returnvalues'][gp['state']['current_individual']] = []
            gp['class']['fitnesstrain'][gp['state']['current_individual']] = np.inf
            gp['class']['fitnessvalid'][gp['state']['current_individual']] = np.inf
            gp['class']['fitnesstest'][gp['state']['current_individual']] = np.inf
            fitness = np.inf
            return fitness, gp

        for cc in range(popNum):
            geneOutputstr2 = geneOutputstr1[cc]
            geneOutputsval2 = geneOutputsval1[cc]
            geneOutputsts2 = geneOutputsts1[cc]
            yptr, probtr = net[cc].predict(geneOutputstr2[0]), net[cc].predict_proba(geneOutputstr2[0])
            ypval, probval = net[cc].predict(geneOutputsval2[0]), net[cc].predict_proba(geneOutputsval2[0])
            ypts, probts = net[cc].predict(geneOutputsts2[0]), net[cc].predict_proba(geneOutputsts2[0])
            weightsgene[cc] = weightsgene[cc]
            gradient[cc] = gradient[cc]
            probtrain[cc] = probtr
            probvalid[cc] = probval
            probtest[cc] = probts
            losstrain[cc] = logloss(probtr, ybinarytr)
            lossvalid[cc] = logloss(probval, ybinaryval)
            losstest[cc] = logloss(probts, ybinaryts)
            yptrain[cc] = yptr
            ypvalid[cc] = ypval
            yptest[cc] = ypts

        if popNum == 1:
            weightsindiv = 1
            gradient_en = 0
            fitnesstrain = losstrain
            fitnessvalid = lossvalid
            fitnesstest = losstest
            probtrain_en = probtrain[0]
            probvalid_en = probvalid[0]
            probtest_en = probtest[0]
            losstrain_en = losstrain
            lossvalid_en = lossvalid
            losstest_en = losstest
            yptrain_en = yptrain[0]
            ypvalid_en = ypvalid[0]
            yptest_en = yptest[0]
            individual = evalstr1
        else:
            indivProbtr = [None] * clsNum
            indivProbval = [None] * clsNum
            indivProbts = [None] * clsNum
            for kk in range(clsNum):
                indivProbtr[kk] = np.zeros((numDatatr, popNum))
                indivProbval[kk] = np.zeros((numDataval, popNum))
                indivProbts[kk] = np.zeros((numDatats, popNum))
                for gg in range(popNum):
                    indivProbtr[kk][:, gg] = probtrain[gg][:, kk]
                    indivProbval[kk][:, gg] = probvalid[gg][:, kk]
                    indivProbts[kk][:, gg] = probtest[gg][:, kk]

            if ensemble_method == 'linear':
                try:
                    w, grad, lossval = gradient_descent_linear(
                        indivProbtr, indivProbval, ybinarytr, ybinaryval,
                        initial_learning_rate, epsilon, lambda_L1, lambda_L2,
                        tolgrad, max_iteration
                    )
                    weightsindiv = w
                    gradient_en = grad
                    fitnessvalid = lossval
                    ensemblefun = linearensemble(indivProbtr)
                except:
                    fitnesstrain = np.inf
                    fitnessvalid = np.inf
                    fitnesstest = np.inf

                if np.isinf(fitnesstrain).any():
                    gp['fitness']['returnvalues'][gp['state']['current_individual']] = []
                    gp['fitness']['fitnesstrain'][gp['state']['current_individual']] = np.inf
                    gp['fitness']['fitnessvalid'][gp['state']['current_individual']] = np.inf
                    gp['fitness']['fitnesstest'][gp['state']['current_individual']] = np.inf
                    fitness = np.inf
                    return fitness, gp

            elif ensemble_method == 'softmax':
                try:
                    w, grad, lossval = gradient_descent_softmax(
                        indivProbtr, indivProbval, ybinarytr, ybinaryval,
                        initial_learning_rate, epsilon, lambda_L1, lambda_L2,
                        tolgrad, max_iteration
                    )
                    weightsindiv = w
                    gradient_en = grad
                    fitnessvalid = lossval
                    ensemblefun = softmax(indivProbtr)
                except:
                    fitnesstrain = np.inf
                    fitnessvalid = np.inf
                    fitnesstest = np.inf

                if np.isinf(fitnesstrain).any():
                    gp['fitness']['returnvalues'][gp['state']['current_individual']] = []
                    gp['fitness']['fitnesstrain'][gp['state']['current_individual']] = np.inf
                    gp['fitness']['fitnessvalid'][gp['state']['current_individual']] = np.inf
                    gp['fitness']['fitnesstest'][gp['state']['current_individual']] = np.inf
                    fitness = np.inf
                    return fitness, gp

                probtrain_en = ensemblefun(indivProbtr, weightsindiv)
                probvalid_en = ensemblefun(indivProbval, weightsindiv)
                probtest_en = ensemblefun(indivProbts, weightsindiv)
                losstrain_en = logloss(probtrain_en, ybinarytr)
                lossvalid_en = logloss(probvalid_en, ybinaryval)
                losstest_en = logloss(probtest_en, ybinaryts)
                fitnesstrain = losstrain_en
                fitnessvalid = lossvalid_en
                fitnesstest = losstest_en
                yptrain_en = argmax(probtrain_en)
                ypvalid_en = argmax(probvalid_en)
                yptest_en = argmax(probtest_en)

            elif ensemble_method == 'ann':
                probanntr = np.zeros((numDatatr, clsNum * popNum))
                probannval = np.zeros((numDataval, clsNum * popNum))
                probannts = np.zeros((numDatats, clsNum * popNum))
                for kk1 in range(clsNum):
                    probanntr[:, (kk1-1)*popNum:kk1*popNum] = indivProbtr[kk1]
                    probannval[:, (kk1-1)*popNum:kk1*popNum] = indivProbval[kk1]
                    probannts[:, (kk1-1)*popNum:kk1*popNum] = indivProbts[kk1]

                try:
                    Mdlen = MLPClassifier(hidden_layer_sizes=gp['runcontrol']['layer'])
                    Mdlen.fit(probanntr, ytr)
                    gp['class']['ensemblemdl'][gp['state']['current_individual']] = Mdlen
                    weightsindiv = Mdlen.coefs_
                    gradient_en = Mdlen.loss_curve_
                    fitnessvalid = Mdlen.loss_
                except:
                    fitnesstrain = np.inf
                    fitnessvalid = np.inf
                    fitnesstest = np.inf

                if np.isinf(fitnesstrain).any():
                    gp['fitness']['returnvalues'][gp['state']['current_individual']] = []
                    gp['fitness']['fitnesstrain'][gp['state']['current_individual']] = np.inf
                    gp['fitness']['fitnessvalid'][gp['state']['current_individual']] = np.inf
                    gp['fitness']['fitnesstest'][gp['state']['current_individual']] = np.inf
                    fitness = np.inf
                    return fitness, gp

                yptr, probtr = Mdlen.predict(probanntr), Mdlen.predict_proba(probanntr)
                ypval, probval = Mdlen.predict(probannval), Mdlen.predict_proba(probannval)
                ypts, probts = Mdlen.predict(probannts), Mdlen.predict_proba(probannts)
                probtrain_en = probtr
                probvalid_en = probval
                probtest_en = probts
                losstrain_en = logloss(probtr, ybinarytr)
                lossvalid_en = logloss(probval, ybinaryval)
                losstest_en = logloss(probts, ybinaryts)
                fitnesstrain = losstrain_en
                fitnessvalid = lossvalid_en
                fitnesstest = losstest_en
                yptrain_en = yptr
                ypvalid_en = ypval
                yptest_en = ypts

        for ii in range(popNum):
            gp['class']['pop'][gp['state']['current_individual'], ii] = evalstr1[ii]
            gp['class']['net'][gp['state']['current_individual'], ii] = net[ii]
            gp['class']['weight_genes'][gp['state']['current_individual'], ii][gp['state']['gen']] = weightsgene[ii]
            gp['class']['gradient_indiv'][gp['state']['current_individual'], ii][gp['state']['gen']] = gradient[ii]
            gp['class']['lossindiv_train'][gp['state']['current_individual'], ii][gp['state']['gen']] = losstrain[ii]
            gp['class']['lossindiv_validation'][gp['state']['current_individual'], ii][gp['state']['gen']] = lossvalid[ii]
            gp['class']['lossindiv_test'][gp['state']['current_individual'], ii][gp['state']['gen']] = losstest[ii]
            gp['class']['geneOutput_train'][gp['state']['current_individual'], ii] = geneOutputstr1[ii]
            gp['class']['geneOutput_validation'][gp['state']['current_individual'], ii] = geneOutputsval1[ii]
            gp['class']['geneOutput_test'][gp['state']['current_individual'], ii] = geneOutputsts1[ii]
            gp['class']['probindiv_train'][gp['state']['current_individual'], ii] = probtrain[ii]
            gp['class']['probindiv_validation'][gp['state']['current_individual'], ii] = probvalid[ii]
            gp['class']['probindiv_test'][gp['state']['current_individual'], ii] = probtest[ii]
            gp['class']['ypindiv_train'][gp['state']['current_individual'], ii] = yptrain[ii]
            gp['class']['ypindiv_valid'][gp['state']['current_individual'], ii] = ypvalid[ii]
            gp['class']['ypindiv_test'][gp['state']['current_individual'], ii] = yptest[ii]

        gp['class']['weight_ensemble'][gp['state']['current_individual'], 0][gp['state']['gen']] = weightsindiv
        gp['class']['gradient_ensemble'][gp['state']['current_individual'], 0][gp['state']['gen']] = gradient_en
        gp['class']['fitness_train_ensemble'][gp['state']['current_individual'], 0] = fitnesstrain
        gp['class']['fitness_validation_ensemble'][gp['state']['current_individual'], 0] = fitnessvalid
        gp['class']['fitness_test_ensemble'][gp['state']['current_individual'], 0] = fitnesstest
        gp['class']['prob_train_ensemble'][gp['state']['current_individual'], 0] = probtrain_en
        gp['class']['prob_validation_ensemble'][gp['state']['current_individual'], 0] = probvalid_en
        gp['class']['prob_test_ensemble'][gp['state']['current_individual'], 0] = probtest_en
        gp['class']['loss_train_ensemble'][gp['state']['current_individual'], 0][gp['state']['gen']] = losstrain_en
        gp['class']['loss_valid_ensemble'][gp['state']['current_individual'], 0][gp['state']['gen']] = lossvalid_en
        gp['class']['loss_test_ensemble'][gp['state']['current_individual'], 0][gp['state']['gen']] = losstest_en
        gp['class']['yp_train_ensemble'][gp['state']['current_individual'], 0] = yptrain_en
        gp['class']['yp_valid_ensemble'][gp['state']['current_individual'], 0] = ypvalid_en
        gp['class']['yp_test_ensemble'][gp['state']['current_individual'], 0] = yptest_en
        gp['class']['fitnessindiv'][gp['state']['current_individual'], :] = lossvalid

        gp['fitness']['returnvalues'][gp['state']['current_individual']] = weightsindiv
    else:
        weightsindiv = gp['fitness']['returnvalues'][gp['state']['current_individual']]

    fitness = gp['class']['fitness_validation_ensemble'][gp['state']['current_individual'], 0]
    return fitness, gp
