import numpy as np, pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict

class Dataloader:    
    def vocab(self, Df, subseq_size=5):
        V = {}
        counter = -1
        for seq in Df:
            for i in range(len(seq[0])-subseq_size+1):
                key = seq[0][i:i+subseq_size]
                if key not in V:
                    counter += 1
                    V[key] = counter
        self.vocabulary = V
        return V
    
    def transform1(self, Df, V, subseq_size = 8, count=False):
        NewDf = []
        n = len(V)
        for seq in Df:
            counts = defaultdict(lambda: 0.0)
            vector = [0]*n
            if not count:
                for j in range(len(seq[0])-subseq_size+1):
                    key = seq[0][j:j+subseq_size]
                    vector[V[key]] = 1.
            else:
                for i in range(len(seq[0])-subseq_size+1):
                    key = seq[0][i:i+subseq_size]
                    counts[key] += 1.
                for j in range(len(seq[0])-subseq_size+1):
                    key = seq[0][j:j+subseq_size]
                    vector[V[key]] = counts[key]/10.
            NewDf.append(vector)
        return np.array(NewDf)
        
    def transform2(self, Xmat):
        Data = {'seq': []}
        for seq in Xmat.values:
            seq = seq[0].split()
            seq = list(map(lambda x: float(x), seq))
            Data['seq'].append(seq)
        return np.array(Data['seq'])
    
    def load_data(self, dirXtr, dirXtr_mat, dirX_te, dirXte_mat, dir_y, use_data_mat=False, subseq_size=5, count=True):
        X, y, Xte = pd.read_csv(dirXtr), pd.read_csv(dir_y), pd.read_csv(dirX_te)
        Xtr, Xtest = pd.read_csv(dirXtr_mat), pd.read_csv(dirXte_mat)

        Df = X.values[:, 1:]
        Dfe = Xte.values[:, 1:]
        D = np.concatenate([Df, Dfe])

        V = self.vocab(D, subseq_size=subseq_size)
        Df = self.transform1(Df, V, subseq_size=subseq_size, count=count)
        Dfe = self.transform1(Dfe, V, subseq_size=subseq_size, count=count)
        
        if use_data_mat:
            Xtr_mat = self.transform2(Xtr)
            Xte_mat = self.transform2(Xtest)

            Df = np.concatenate([Df, Xtr_mat], axis=1)
            Dfe = np.concatenate([Dfe, Xte_mat], axis=1)
            
        y = y.values[:,1:].flatten()
        y[y==0]=-1.
        
        Df = np.concatenate([Df, y.reshape(-1,1)], axis=1)
        Dfe  = np.concatenate([Dfe, np.zeros(len(Dfe)).reshape(-1,1)], axis=1)

        return np.array(Df, dtype=float), np.array(Dfe, dtype=float)
    
    def split_data(self, Df, train_size=0.8):
        Rand = np.random.permutation(range(len(Df)))
        Df = Df[Rand]
        Xtrain = Df[:int(train_size*len(Df)), :-1]
        ytrain = Df[:int(train_size*len(Df)), -1]
        Xval = Df[int(train_size*len(Df)):, :-1]
        yval = Df[int(train_size*len(Df)):, -1]
        return np.array(Xtrain), np.array(ytrain), np.array(Xval), np.array(yval)

class CrossValidation:
    def __init__(self, model_name='svm', k = 5):
        self.k = k
        assert model_name in ['svm', 'logistic_regression'], 'the model name should be svm or logistic_regression'
        self.model_name = model_name
        
    def Kfold(self, X, y):
        k = self.k
        Rand = np.random.permutation(range(len(X)))
        X = X[Rand]
        y = y[Rand]
        Folds = []
        for i in range(k):
            Folds.append([X[i*(len(X)//k):(i+1)*(len(X)//k)], y[i*(len(y)//k):(i+1)*(len(y)//k)]])
        self.Folds = Folds
        return self
    
    def crossvalidate(self):
        Folds = self.Folds
        if self.model_name == 'svm':
            if SVM.use_kernel: # Look for the best value of the degree
                best_degree, best_val_acc = None, 0.0
                degrees = np.linspace(1, 3, 8)
                for j, degree in enumerate(degrees):
                    print('*'*50)
                    print('Progress: {}/{}'.format(j+1, len(degrees)))
                    print('*'*50)
                    Acc = []
                    SVM.degree = degree
                    for i in range(self.k):
                        Xv, yv = Folds[i]
                        Rest = [Folds[l] for l in range(self.k) if l != i]
                        concatX, concaty = Rest[0][0], Rest[0][1]
                        for l in range(1,len(Rest)):
                            concatX, concaty = np.vstack([concatX, Rest[l][0]]), np.hstack([concaty, Rest[l][1]])
                        Xt, yt = concatX, concaty
                        SVM.fit_kernel(Xt, yt)
                        y_pred = SVM.Kernel_predict(Xv)
                        Acc.append(SVM.compute_accuracy(y_pred, yv))
                    acc = np.mean(Acc)
                    print('Val acc: ', acc)
                    if acc > best_val_acc:
                        best_val_acc = acc
                        best_degree = degree
                print('Best val acc: ', best_val_acc)
                return best_degree
            else: # Look for te best value of C for a simple SVM
                best_C, best_val_acc = None, 0.0
                Cvalues = 10**np.linspace(-1, 3, 5)
                for j, C in enumerate(Cvalues):
                    print('*'*50)
                    print('Progress: {}/{}'.format(j+1, len(Cvalues)))
                    print('*'*50)
                    Acc = []
                    SVM.C = C
                    for i in range(self.k):
                        Xv, yv = Folds[i]
                        Rest = [Folds[l] for l in range(self.k) if l != i]
                        concatX, concaty = Rest[0][0], Rest[0][1]
                        for l in range(1,len(Rest)):
                            concatX, concaty = np.vstack([concatX, Rest[l][0]]), np.hstack([concaty, Rest[l][1]])
                        Xt, yt = concatX, concaty
                        SVM.fit_no_kernel(Xt, yt)
                        y_pred = SVM.predict_no_kernel(Xv)
                        Acc.append(SVM.compute_accuracy(y_pred, yv))
                    acc = np.mean(Acc)
                    print('Val acc: ', acc)
                    if acc > best_val_acc:
                        best_val_acc = acc
                        best_C = C
                print('Best val acc: ', best_val_acc)
                return best_C
                               
        else:
            # Logistic regression part
            best_lambd, best_val_acc = None, 0.0
            Lambdas = 10**np.linspace(-1, 1, 5)
            for j,lambd in enumerate(Lambdas):
                print('*'*50)
                print('Progress: {}/{}'.format(j+1, len(Lambdas)))
                print('*'*50)
                LR.lambd = lambd
                Acc = []
                for i in range(self.k):
                    Xv, yv = Folds[i]
                    Rest = [Folds[l] for l in range(self.k) if l != i]
                    concatX, concaty = Rest[0][0], Rest[0][1]
                    for l in range(1,len(Rest)):
                        concatX, concaty = np.vstack([concatX, Rest[l][0]]), np.hstack([concaty, Rest[l][1]])
                    Xt, yt = concatX, concaty
                    LR.fit(yt, Xt, max_iter=10)
                    y_pred = LR.predict(Xv)
                    Acc.append(LR.compute_accuracy(y_pred, yv))
                acc = np.mean(Acc)
                print('Val acc: ', acc)
                if acc > best_val_acc:
                    best_val_acc = acc
                    best_lambd = lambd
            print('Best val acc:', best_val_acc)
            return best_lambd