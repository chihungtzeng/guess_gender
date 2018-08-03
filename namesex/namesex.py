# -*- coding: utf-8 -*-
import csv
import operator
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
import pkg_resources, pickle, gzip

class namesex:
    def __init__(self, n_jobs=-1, n_estimators = 500, loadmodel = True, \
                   w2v_filename = \
                   pkg_resources.resource_filename('namesex', 'w2v_dictvec_sg_s100i20.pkl')):
        import platform
        from pathlib import Path


        self.gname_count = dict()
        self.gnameug_count = dict()
        self.feat_dict = dict()
        self.num_feature = 0
        self.num_gname = 0
        self.lrmodelintcp = None
        self.lrmodelcoef = None
        self.w2v_dictvec = None
        #mean of diverge
        self.w2v_pooling = "diverge"
        self.rfmodel = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)

        #load w2v data
        with open(w2v_filename, "rb") as f:
            self.w2v_dictvec = pickle.load(f)
        #firstchar = self.w2v_dictvec.keys()[0]
        firstchar = next(iter(self.w2v_dictvec.keys()))
        self.w2v_vecsize = len(self.w2v_dictvec[firstchar])


    def gen_feature_id(self, namevec, min_ugram = 2, min_gname = 2):
        for gname in namevec:
            if gname in self.gname_count:
                self.gname_count[gname] += 1
            else:
                self.gname_count[gname] = 1
            for achar in gname:
                if achar in self.gnameug_count:
                    self.gnameug_count[achar] += 1
                else:
                    self.gnameug_count[achar] = 1

        gname_sorted = sorted(self.gname_count.items(), key=operator.itemgetter(1), reverse=True)
        gnameug_sorted = sorted(self.gnameug_count.items(), key=operator.itemgetter(1), reverse=True)
        #print("Top 20 given names: {}".format(gname_sorted[0:20]))
        #for apair in gname_sorted[0:20]:
        #    print("{}, female-male: {}".format(apair, gname_sex_count[apair[0]]))

        fid = 0
        for apair in gname_sorted:
            if apair[0] in self.feat_dict:
                print("Error! {} already exists".format(apair[0]))
            else:
                if apair[1] >= min_gname:
                    self.feat_dict[apair[0]] = fid
                    fid += 1
        # number of gnames included
        self.num_gname = fid

        for apair in gnameug_sorted:
            if apair[1] >= min_ugram:
                if apair[0] in self.feat_dict:
                    #print("Warning! {} already exists".format(apair[0]))
                    pass
                else:
                    self.feat_dict[apair[0]] = fid
                    fid += 1
        # add "_Other_Value_"
        self.feat_dict["_Other_Value_"] = fid
        self.num_feature = fid + 1

    def feature_coding(self, aname):
        tmpfeat = list()
        has_other = 0
        if aname in self.feat_dict:
            tmpfeat.append(self.feat_dict[aname])

        for achar in aname:
            if achar in self.feat_dict:
                tmpfeat.append(self.feat_dict[achar])

        if len(tmpfeat) == 0:
            tmpfeat.append(self.feat_dict["_Other_Value_"])
        return tmpfeat

    #generate unigram and gname feature array
    def gen_feat_array(self, namevec):
        #name_given_int = list()
        x_array = np.zeros((len(namevec), self.num_feature), "int")
        for id, aname in enumerate(namevec):
            #name_given_int.append(self.feature_coding(aname))
            x_array[id, self.feature_coding(aname)] = 1
        return x_array
    def gen_feat_array_w2v(self, namevec):
        x_train = self.gen_feat_array(namevec)
        xw2v_train1, xw2v_train2 = self.extract_w2v(namevec)

        if self.w2v_pooling == "mean":
            x2_train = np.concatenate((x_train, xw2v_train1), axis=1)
        else:
            x2_train = np.concatenate((x_train, xw2v_train2), axis=1)
        return x2_train

    # add w2v features.
    # w2vmodel1.vector_size
    # note: use global variables.
    def extract_w2v(self, namearray):
        xw2v1 = np.zeros((len(namearray), self.w2v_vecsize), "float")
        xw2v2 = np.zeros((len(namearray), self.w2v_vecsize), "float")
        for i, aname in enumerate(namearray):
            # preserve the mean
            vec_mean = np.zeros((self.w2v_vecsize), "float")
            # want to preserve the part that is farest from zero for each dimension
            # positive part
            vec_diverge1 = np.zeros((self.w2v_vecsize), "float")
            # negative part
            vec_diverge0 = np.zeros((self.w2v_vecsize), "float")
            nc1 = 0
            for achar in aname:
                try:
                    # tmp = w2vmodel1[achar]
                    tmp = self.w2v_dictvec[achar]
                    tmp = tmp / np.linalg.norm(tmp)
                    vec_mean = vec_mean + tmp
                    nc1 += 1
                    # divergent
                    ind1 = tmp >= 0
                    tmp1 = tmp * ind1
                    tmp0 = tmp * (1 - ind1)
                    vec_diverge1 = np.max(np.vstack((vec_diverge1, tmp1)), axis=0)
                    vec_diverge0 = np.min(np.vstack((vec_diverge0, tmp0)), axis=0)
                except:
                    #print("{} not in w2v model, skip".format(achar))
                    pass

                if nc1 > 1:
                    vec_mean = vec_mean / nc1
                vec_diverge = vec_diverge1 + vec_diverge0
                xw2v1[i] = vec_mean
                xw2v2[i] = vec_diverge
        return xw2v1, xw2v2

    def train(self, namevec, sexvec, c2 = 10):
        #tran logistic regression (unigram and given names) and random forest (with word2vec features)
        #print("Training random forest")

        self.gen_feature_id(namevec)
        y_train = np.asarray(sexvec)

        logreg = linear_model.LogisticRegression(C=c2)
        x_array = self.gen_feat_array(namevec)
        logreg.fit(x_array, y_train)
        #self.lrmodel = logreg
        self.lrmodelcoef = np.transpose(logreg.coef_)
        self.lrmodelintcp = logreg.intercept_

        x2_train = self.gen_feat_array_w2v(namevec)
        self.rfmodel.fit(x2_train, y_train)


if __name__ == "__main__":
    #load data
    f = open('data/namesex_data_v2.csv', 'r', newline='', encoding = 'utf8')
    mydata = csv.DictReader(f)
    sexlist = []
    namelist = []
    foldlist = []
    for arow in mydata:
        sexlist.append(int(arow['sex'].strip()))
        gname = arow['gname'].strip()
        namelist.append(gname)
        foldlist.append(int(arow['fold'].strip()))

    sexlist = np.asarray(sexlist)
    namelist = np.asarray(namelist)
    foldlist = np.asarray(foldlist)


    np.random.seed(1034)
    ns = namesex(loadmodel=False)
    print("Training models (logistic reg and random forest)")
    ns.train(namelist, sexlist)
    print("training completed.")
