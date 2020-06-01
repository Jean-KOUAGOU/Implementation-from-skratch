import numpy as np, pandas as pd
from Data_Dataloader_CrossVal.dataloader import Dataloader, CrossValidation
from Models.models import MySVM, LogisticRegression
DataLoader = Dataloader()
Xtrf, Xtrmatf, Xtef, Xtematf, Ytrf = './Data_Dataloader_CrossVal/Xtr.csv', './Data_Dataloader_CrossVal/Xtr_mat100.csv', './Data_Dataloader_CrossVal/Xte.csv', './Data_Dataloader_CrossVal/Xte_mat100.csv', './Data_Dataloader_CrossVal/Ytr.csv'
Df_tr, Df_te = DataLoader.load_data(Xtrf, Xtrmatf, Xtef, Xtematf, Ytrf, use_data_mat=True, subseq_size=6, count=True)
np.random.seed(101)
X_train, y_train, X_val, y_val = DataLoader.split_data(Df_tr, train_size=0.9)
X_test, _, _, _ = DataLoader.split_data(Df_te,train_size=1.)

# Uncomment the following to run the logistic regression

# LR = LogisticRegression()
# LR.lambd = 0.0001
# LR.fit(y_train, X_train, max_iter=20)
# pred=LR.predict(X_val)
# print('val acc:', LR.compute_accuracy(y_val, pred))
# pred = LR.predict(X_train)
# print('train acc:', LR.compute_accuracy(y_train, pred))
# Yte = LR.predict(X_test)
# Yte[Yte==-1]=0

SVM = MySVM()
SVM.kernels = ['poly', 'gauss']
SVM.combine_kernel = 'prod'
SVM.C=1
SVM.std = 35.
SVM.degree=2.14
SVM.fit_kernel(X_train, y_train)
y_pred = SVM.Kernel_predict(X_train)
print('train accuracy:', SVM.compute_accuracy(y_pred, y_train))
y_pred = SVM.Kernel_predict(X_val)
print('val accuracy:', SVM.compute_accuracy(y_pred, y_val))

Yte = SVM.Kernel_predict(X_test)
Yte[Yte==-1]=0
Yte = list(map(lambda x: int(x), Yte))
submission = {"Id":list(range(len(Yte))), "Bound": Yte}
submission_df = pd.DataFrame(submission)
submission_df.to_csv('Submission.csv',columns=["Id","Bound"], index=False)
print()
print("Please find the generated submission file (Submission.csv) in the main repository")
print()
