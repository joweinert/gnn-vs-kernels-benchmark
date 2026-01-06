from grakel.kernels import WeisfeilerLehman, VertexHistogram, GraphletSampling
from grakel import Graph
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class KernelSVC:
    """
    Wrapper for a Kernel-based Classifier using GraKeL and Scikit-Learn.
    Architecture:
      Input (Graphs) -> WL Kernel (computes Kernel Matrix K) -> Precomputed Kernel SVM
    """
    def __init__(self, kernel_type='WL', C=1.0, n_iter=5, k=5, n_samples=100, normalize=True):
        self.kernel_type = kernel_type
        self.C = C
        self.normalize = normalize
        
        # kernel init
        if self.kernel_type == 'WL':
            self.n_iter = int(n_iter)
            self.gk = WeisfeilerLehman(n_iter=self.n_iter, base_graph_kernel=VertexHistogram, normalize=self.normalize)
        elif self.kernel_type == 'GS':
            self.k = int(k)
            self.n_samples = int(n_samples)
            # GraphletSampling expects sampling dict, e.g. {'n_samples': 100}
            self.gk = GraphletSampling(k=self.k, sampling={'n_samples': self.n_samples}, normalize=self.normalize)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
            
        self.svc = SVC(kernel='precomputed', C=self.C)
        self.model = None

    def get_metrics(self, X_test, y_test):
        """Returns dict of Acc, F1, AUC"""
        K_test = self.gk.transform(X_test)        
        y_pred = self.svc.predict(K_test)
        y_score = self.svc.decision_function(K_test) 
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        try:
            auc = roc_auc_score(y_test, y_score)
        except ValueError:
            auc = 0.0
            
        return acc, f1, auc

    def fit(self, X_train, y_train):
        """
        X_train: List of NetworkX graphs or Grakel Graphs
        y_train: Labels
        """
        # computes Kernel Matrix K_train
        # Grakel expects list of iterables or Grakel Graph objects
        # X_train is a list of NetworkX graphs 
        
        self.gk.fit(X_train)
        K_train = self.gk.transform(X_train)
        
        # svm fit
        self.svc.fit(K_train, y_train)
        return self

    def predict(self, X_test):
        # computes Kernel Matrix K_test (similarity between X_test and X_train support vectors)
        K_test = self.gk.transform(X_test)
        
        # svm predict
        return self.svc.predict(K_test)
